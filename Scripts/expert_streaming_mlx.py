"""Opt-in exact-weight expert streaming for the two pinned fleet MoE models.

Uses MLX's existing quantized Metal matmuls, with bounded pread expert loading.
This trades CPU/GPU synchronization and SSD I/O for lower weight residency.
"""
import logging
from pathlib import Path
import threading
import weakref

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.activations import swiglu
from mlx_lm.utils import _get_classes, load_config, load_tokenizer
from omlx.utils.metal_sync import _sync_and_clear_cache

from expert_streaming import TensorFiles, LayerExpertCache, EXPERT, layout

logger = logging.getLogger(__name__)
NUMPY_TYPES = {'U32': '<u4', 'F32': '<f4', 'F16': '<f2', 'BF16': '<u2', 'I32': '<i4', 'I64': '<i8'}


def tensor_array(payload):
    data, dtype, shape = payload
    value = mx.array(np.frombuffer(data, dtype=NUMPY_TYPES[dtype]).reshape(shape))
    if dtype == 'BF16':
        value = value.view(mx.bfloat16)
    mx.eval(value)
    return value


class ExpertStore:
    def __init__(self, files, config, cache_bytes):
        self.files, self.config = files, config
        self.lock = threading.RLock()
        self.closed = False
        self.profile = layout(files, config, cache_bytes)
        self.cache = LayerExpertCache(cache_bytes, self.profile['layers'])

    def expert(self, prefix, index):
        names = [f'{prefix}.{projection}.{field}'
                 for projection in ('gate_proj', 'up_proj', 'down_proj')
                 for field in ('weight', 'scales', 'biases')]
        size = sum(self.files.tensors[n].size // self.files.tensors[n].shape[0] for n in names)
        return self.cache.get((prefix, index), size,
                              lambda: {n: tensor_array(self.files.read(n, index)) for n in names})

    def close(self):
        with self.lock:
            self.cache.clear()
            self.files.close()
            self.closed = True


class StreamedSwitchGLU(nn.Module):
    def __init__(self, store, prefix, chunk_rows=16):
        super().__init__()
        self.store, self.prefix, self.chunk_rows = store, prefix, chunk_rows
        if not 1 <= chunk_rows <= 64:
            raise ValueError('Streamed expert row chunk must be 1–64')

    def __call__(self, x, indices):
        # Freeze routed indices on the host. No route is dropped or approximated.
        with self.store.lock:
            if self.store.closed:
                raise RuntimeError('Expert files have been closed')
            mx.eval(x, indices)
            shape, hidden = indices.shape, x.shape[-1]
            count, top_k = indices.size // shape[-1], shape[-1]
            if x.size // hidden != count:
                raise ValueError('Expert input and routed indices disagree')
            flat_x, flat_indices = x.reshape(count, hidden), indices.reshape(count, top_k)
            chunks = []
            for start in range(0, count, self.chunk_rows):
                end = min(count, start + self.chunk_rows)
                selected = flat_indices[start:end].tolist()
                positions = {}
                for row, choices in enumerate(selected):
                    for choice, expert in enumerate(choices):
                        positions.setdefault(expert, []).append(row * top_k + choice)
                output = mx.zeros(((end - start) * top_k, hidden), dtype=x.dtype)
                for expert, slots in positions.items():
                    bank = self.store.expert(self.prefix, expert)
                    loc = mx.array(slots, dtype=mx.int32)
                    rows = mx.array([slot // top_k for slot in slots], dtype=mx.int32)
                    inputs = flat_x[start:end][rows]

                    def project(name, value):
                        key = f'{self.prefix}.{name}'
                        global_q = self.store.config['quantization']
                        q = global_q.get(key, global_q)
                        return mx.quantized_matmul(value, bank[key + '.weight'],
                            bank[key + '.scales'], bank[key + '.biases'], transpose=True,
                            group_size=q['group_size'], bits=4, mode='affine')

                    gate, up = project('gate_proj', inputs), project('up_proj', inputs)
                    result = project('down_proj', swiglu(gate, up))
                    output = output.at[loc].add(result)
                    # Drain this graph before an eviction can release expert
                    # buffers. Old experts cannot accumulate through lazy outputs.
                    mx.eval(output)
                    del bank, result, gate, up, inputs
                chunks.append(output.reshape(end - start, top_k, hidden))
                # Preserve oMLX's high allocator cache limit (M4 safety). Free
                # unused staging buffers only at a synchronized boundary.
                _sync_and_clear_cache(mx.default_stream(mx.default_device()))
            if not chunks:
                return mx.zeros((*shape, hidden), dtype=x.dtype)
            result = mx.concatenate(chunks, axis=0).reshape(*shape, hidden)
            mx.eval(result)
            return result


def load_streamed_model(directory, cache_bytes=256 * 1024**2, chunk_rows=16):
    directory = Path(directory)
    config = load_config(directory)
    files = TensorFiles(directory)
    store = None
    try:
        store = ExpertStore(files, config, cache_bytes)
        model_class, args_class = _get_classes(config)
        model = model_class(args_class.from_dict(config))
        # The standard constructors are lazy. Remove routed parameter graphs
        # BEFORE quantization, loading, or evaluation could materialize them.
        for prefix in store.profile['layers']:
            layer = int(prefix.split('.')[2])
            model.layers[layer].mlp.switch_mlp = StreamedSwitchGLU(store, prefix, chunk_rows)
        weights = {name: tensor_array(files.read(name)) for name in files.tensors if not EXPERT.fullmatch(name)}
        weights = model.sanitize(weights)
        quant = config['quantization']

        def predicate(path, module):
            if not hasattr(module, 'to_quantized') or f'{path}.scales' not in weights:
                return False
            return quant.get(path, True)

        nn.quantize(model, group_size=quant['group_size'], bits=quant['bits'],
                    mode=quant.get('mode', 'affine'), class_predicate=predicate)
        model.eval()
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        # Finalization follows the engine/model lifetime; no global array cache.
        weakref.finalize(model, store.close)
        logger.info('Expert streaming loaded %s: core=%d bytes, cache cap=%d bytes, streamed=%d bytes',
                    directory.name, store.profile['resident_bytes'], cache_bytes, store.profile['expert_bytes'])
        return model, config, store
    except BaseException:
        if store is not None:
            store.close()
        else:
            files.close()
        raise


def install_loader(streaming):
    """Intercept only explicitly configured local models; all others delegate.

    No installed package is edited. The hook lives only for this server process.
    oMLX imports this helper at engine load time on its shared MLX executor.
    """
    from omlx.utils import model_loading
    original = model_loading.lm_load_compat
    allowed = {str(Path(path).resolve()): options for path, options in streaming.items()}

    def load(path_or_repo, *, trust_remote_code=False, **kwargs):
        options = allowed.get(str(Path(path_or_repo).resolve()))
        if options is None:
            return original(path_or_repo, trust_remote_code=trust_remote_code, **kwargs)
        if trust_remote_code or set(kwargs) - {'tokenizer_config'}:
            raise ValueError('Unsupported options for the streamed model loader')
        model_loading.preflight_text_remote_code(path_or_repo,
            tokenizer_config=kwargs.get('tokenizer_config'), trust_remote_code=False)
        model, config, store = load_streamed_model(path_or_repo, options['cache_bytes'], options['chunk_rows'])
        try:
            tokenizer_config = dict(kwargs.get('tokenizer_config') or {})
            tokenizer_config['trust_remote_code'] = False
            tokenizer = load_tokenizer(Path(path_or_repo), tokenizer_config,
                                       eos_token_ids=config.get('eos_token_id'))
        except BaseException:
            store.close()
            raise
        return model, tokenizer

    model_loading.lm_load_compat = load
    return original
