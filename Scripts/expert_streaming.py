"""Validated safetensors ranges and bounded expert-cache planning (no MLX import)."""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import struct
import threading

DTYPES = {'U32': 4, 'F32': 4, 'F16': 2, 'BF16': 2, 'I32': 4, 'I64': 8}
EXPERT = re.compile(r'^(model\.layers\.(\d+)\.mlp\.switch_mlp)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
SUPPORTED = {'qwen3_moe', 'glm4_moe_lite'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate safetensors key: {key}')
        result[key] = value
    return result


def read_exact(fd, size, offset):
    chunks = []
    while size:
        data = os.pread(fd, min(size, 1024 * 1024), offset)
        if not data:
            raise ValueError('Truncated safetensors payload')
        chunks.append(data)
        size -= len(data)
        offset += len(data)
    return b''.join(chunks)


@dataclass(frozen=True)
class TensorRange:
    file: Path
    dtype: str
    shape: tuple
    offset: int
    size: int


class TensorFiles:
    """Only opens descriptors and headers; never maps an entire checkpoint."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.descriptors = {}
        self.tensors = {}
        self.bytes_read = 0
        self.read_counter_lock = threading.Lock()
        try:
            if self.directory.is_symlink():
                raise ValueError('Model directory cannot be a symlink')
            paths = sorted(self.directory.glob('model*.safetensors'))
            if not paths:
                raise ValueError('No completed safetensors files')
            for path in paths:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                self.descriptors[path] = fd
                file_size = os.fstat(fd).st_size
                length = struct.unpack('<Q', read_exact(fd, 8, 0))[0]
                if not 2 <= length <= min(16 * 1024**2, file_size - 8):
                    raise ValueError('Invalid safetensors header length')
                header = json.loads(read_exact(fd, length, 8), object_pairs_hook=unique_object)
                spans = []
                for name, tensor in header.items():
                    if name == '__metadata__':
                        continue
                    dtype, shape = tensor['dtype'], tensor['shape']
                    start, end = tensor['data_offsets']
                    if (dtype not in DTYPES or not isinstance(shape, list)
                            or any(type(n) is not int or n < 0 for n in shape)
                            or type(start) is not int or type(end) is not int
                            or start < 0 or end < start or 8 + length + end > file_size
                            or end - start != math.prod(shape) * DTYPES[dtype]):
                        raise ValueError(f'Invalid tensor layout: {name}')
                    if name in self.tensors:
                        raise ValueError(f'Duplicate tensor across shards: {name}')
                    self.tensors[name] = TensorRange(path, dtype, tuple(shape), 8 + length + start, end - start)
                    spans.append((start, end))
                cursor = 0
                for start, end in sorted(spans):
                    if start != cursor:
                        raise ValueError('Overlapping tensors or gaps in safetensors payload')
                    cursor = end
                if cursor != file_size - 8 - length:
                    raise ValueError('Unindexed safetensors payload')
        except BaseException:
            self.close()
            raise

    def read(self, name, expert=None):
        tensor = self.tensors[name]
        offset, size, shape = tensor.offset, tensor.size, tensor.shape
        if expert is not None:
            if not shape or type(expert) is not int or not 0 <= expert < shape[0]:
                raise ValueError('Expert index outside the tensor')
            size //= shape[0]
            offset += expert * size
            shape = shape[1:]
        data = read_exact(self.descriptors[tensor.file], size, offset)
        with self.read_counter_lock:
            self.bytes_read += size
        return data, tensor.dtype, shape

    def close(self):
        for fd in self.descriptors.values():
            os.close(fd)
        self.descriptors.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def layout(files, config, cache_bytes):
    if config.get('model_type') not in SUPPORTED:
        raise ValueError('Expert streaming supports only qwen3_moe and glm4_moe_lite')
    if config.get('model_file') or config.get('quantize_activations'):
        raise ValueError('Custom model code and activation quantization are not supported')
    quant = config.get('quantization', {})
    if quant.get('bits') != 4 or quant.get('mode', 'affine') != 'affine':
        raise ValueError('Expert streaming requires the pinned affine 4-bit layout')
    experts = config.get('num_experts', config.get('n_routed_experts'))
    hidden, intermediate = config['hidden_size'], config['moe_intermediate_size']
    if not isinstance(experts, int) or experts < 1 or cache_bytes < 0:
        raise ValueError('Invalid expert count or cache budget')
    layers = config['num_hidden_layers']
    groups = {}
    resident = 0
    for name, tensor in files.tensors.items():
        match = EXPERT.fullmatch(name)
        if match:
            prefix, layer, projection, field = match.groups()
            if int(layer) >= layers:
                raise ValueError('Unexpected prediction-layer expert bank')
            groups.setdefault(prefix, {})[(projection, field)] = tensor
        elif '.experts.' in name or '.switch_mlp.' in name:
            raise ValueError(f'Unsupported expert layout: {name}')
        else:
            resident += tensor.size
    if config['model_type'] == 'qwen3_moe':
        expected = {f'model.layers.{i}.mlp.switch_mlp' for i in range(layers)
                    if i not in config.get('mlp_only_layers', [])
                    and (i + 1) % config['decoder_sparse_step'] == 0}
    else:
        if config.get('moe_layer_freq', 1) != 1:
            raise ValueError('Unsupported GLM expert-layer frequency')
        expected = {f'model.layers.{i}.mlp.switch_mlp'
                    for i in range(config.get('first_k_dense_replace', 1), layers)}
    if not expected or set(groups) != expected:
        raise ValueError('Missing or unexpected routed-expert layers')
    biggest = 0
    for prefix, fields in groups.items():
        if len(fields) != 9:
            raise ValueError(f'Incomplete expert bank: {prefix}')
        total = 0
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            q = quant.get(f'{prefix}.{projection}', quant)
            bits, group_size = q.get('bits'), q.get('group_size')
            if bits != 4 or q.get('mode', 'affine') != 'affine' or group_size not in (32, 64, 128):
                raise ValueError('Unsupported per-projection expert quantization')
            out_dim, in_dim = (hidden, intermediate) if projection == 'down_proj' else (intermediate, hidden)
            for field in ('weight', 'scales', 'biases'):
                tensor = fields[(projection, field)]
                width = in_dim // (8 if field == 'weight' else group_size)
                if in_dim % group_size or tensor.shape != (experts, out_dim, width):
                    raise ValueError(f'Unexpected expert shape: {prefix}.{projection}.{field}')
                if field == 'weight' and tensor.dtype != 'U32' or field != 'weight' and tensor.dtype not in ('F16', 'BF16', 'F32'):
                    raise ValueError('Unsupported expert dtype')
                total += tensor.size // experts
        biggest = max(biggest, total)
    # Core transform allowance, cache, transient expert staging, allocator pool.
    planned = int(resident * 1.25) + cache_bytes + biggest * 2 + 64 * 1024**2
    return {'resident_bytes': resident, 'expert_bytes': sum(t.size for fields in groups.values() for t in fields.values()),
            'cache_bytes': cache_bytes, 'largest_expert_bytes': biggest,
            'planned_weight_bytes': planned, 'layers': sorted(groups)}


def inspect_model(directory, cache_bytes=256 * 1024**2):
    config = json.loads((Path(directory) / 'config.json').read_text())
    with TensorFiles(directory) as files:
        return layout(files, config, cache_bytes)


class ExpertCache:
    """Byte-bounded cache; LFU admission avoids cyclic-scan LRU thrashing."""
    def __init__(self, budget, policy='lru'):
        self.budget, self.bytes = budget, 0
        self.policy = policy
        self.frequency = {}
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key, size, loader):
        self.frequency[key] = self.frequency.get(key, 0) + 1
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][1]
        self.misses += 1
        if self.policy == 'lfu' and self.bytes + size > self.budget:
            if size > self.budget or (self.entries and self.frequency[key] <= min(self.frequency[k] for k in self.entries)):
                return loader()
        while self.entries and self.bytes + size > self.budget:
            if self.policy == 'lfu':
                victim = min(self.entries, key=lambda k: self.frequency[k])
                old_size, _ = self.entries.pop(victim)
            else:
                _, (old_size, _) = self.entries.popitem(last=False)
            self.bytes -= old_size
            self.evictions += 1
        value = loader()
        if size <= self.budget:
            self.entries[key] = (size, value)
            self.bytes += size
        return value

    def resize(self, budget):
        if type(budget) is not int or budget < 0:
            raise ValueError('Cache budget must be a nonnegative integer')
        self.budget = budget
        while self.entries and self.bytes > budget:
            victim = (min(self.entries, key=lambda k: self.frequency[k])
                      if self.policy == 'lfu' else next(iter(self.entries)))
            size, _ = self.entries.pop(victim)
            self.bytes -= size
            self.evictions += 1

    def clear(self):
        self.entries.clear()
        self.frequency.clear()
        self.bytes = 0


class LayerExpertCache:
    """Share one model budget fairly across layers and across subagent requests.

    A model-wide LRU loses early-layer experts before the next token reaches
    them. Per-layer LFU admission keeps frequently reused experts through scans.
    """
    def __init__(self, budget, layers):
        self.budget = budget
        per_layer, remainder = divmod(budget, len(layers))
        self.layers = {name: ExpertCache(per_layer + (i < remainder), 'lfu')
                       for i, name in enumerate(layers)}

    @property
    def bytes(self): return sum(cache.bytes for cache in self.layers.values())
    @property
    def hits(self): return sum(cache.hits for cache in self.layers.values())
    @property
    def misses(self): return sum(cache.misses for cache in self.layers.values())
    @property
    def evictions(self): return sum(cache.evictions for cache in self.layers.values())

    def get(self, key, size, loader):
        return self.layers[key[0]].get(key[1], size, loader)

    def resize(self, budget):
        if type(budget) is not int or budget < 0:
            raise ValueError('Cache budget must be a nonnegative integer')
        self.budget = budget
        per_layer, remainder = divmod(budget, len(self.layers))
        for i, cache in enumerate(self.layers.values()):
            cache.resize(per_layer + (i < remainder))

    def clear(self):
        for cache in self.layers.values(): cache.clear()
