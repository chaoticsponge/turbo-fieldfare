"""Synthetic, small-model numerical checks; no installed checkpoints needed."""
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.utils import _get_classes
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.switch_layers import SwitchGLU

from expert_streaming import TensorFiles
from expert_streaming_mlx import ExpertStore, StreamedSwitchGLU, load_streamed_model, tensor_array


def tiny_config(kind):
    common = dict(model_type=kind, hidden_size=128, intermediate_size=256,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  rms_norm_eps=1e-5, vocab_size=128, tie_word_embeddings=False,
                  max_position_embeddings=1024, rope_theta=10000.0,
                  moe_intermediate_size=128, norm_topk_prob=True,
                  quantization={'group_size':64,'bits':4})
    if kind == 'qwen3_moe':
        common.update(num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
                      mlp_only_layers=[], head_dim=64)
        for i in range(2): common['quantization'][f'model.layers.{i}.mlp.gate'] = {'group_size':64,'bits':8}
    else:
        common.update(n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
                      first_k_dense_replace=1, kv_lora_rank=64, q_lora_rank=64,
                      qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
                      num_nextn_predict_layers=0, num_key_value_heads=2)
    return common


def create_checkpoint(directory, config):
    cls, args = _get_classes(config)
    model = cls(args.from_dict(config))
    q = config['quantization']
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda path, module: q.get(path, hasattr(module, 'to_quantized')))
    model.eval()
    mx.eval(model.parameters())
    (directory/'config.json').write_text(json.dumps(config))
    mx.save_safetensors(str(directory/'model.safetensors'), dict(tree_flatten(model.parameters())))
    return model


class StreamedModelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(11)

    def test_full_prefill_and_cached_continuation_match_both_architectures(self):
        for kind in ('qwen3_moe', 'glm4_moe_lite'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                reference = create_checkpoint(directory, tiny_config(kind))
                # Explicitly fail if the streaming loader tries the full-shard loader.
                with patch.object(mx, 'load', side_effect=AssertionError('Full shard loaded')):
                    streamed, _, store = load_streamed_model(directory, cache_bytes=20000, chunk_rows=2)
                cache_a, cache_b = make_prompt_cache(reference), make_prompt_cache(streamed)
                for tokens in (mx.array([[1,2,3,4,5]]), mx.array([[6]]), mx.array([[7,8]])):
                    a, b = reference(tokens, cache=cache_a), streamed(tokens, cache=cache_b)
                    mx.eval(a,b)
                    difference = mx.max(mx.abs(a-b)).item()
                    self.assertTrue(mx.allclose(a,b,atol=2e-4,rtol=2e-4).item(), (kind,difference))
                self.assertLessEqual(store.cache.bytes, 20000)
                self.assertGreater(store.cache.misses, 0)
                self.assertFalse(any('.switch_mlp.' in key for key,_ in tree_flatten(streamed.parameters())))
                store.close()

    def test_adaptive_growth_and_shrink_preserve_both_models_outputs(self):
        from adaptive_expert_cache import AdaptiveExpertCaches
        for kind in ('qwen3_moe', 'glm4_moe_lite'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                directory=Path(folder)
                reference=create_checkpoint(directory,tiny_config(kind))
                model,_,store=load_streamed_model(directory,cache_bytes=20000,chunk_rows=1)
                now=[0];memory=[(1000000,8000000,10000000)]
                controller=AdaptiveExpertCaches(100000,8000000,lambda:memory[0],
                    clock=lambda:now[0],interval=2,step_bytes=20000)
                controller.register(store,kind,10000,100000)
                store.adaptive=controller
                tokens=mx.array([[i%100 for i in range(64)]])
                expected=reference(tokens);mx.eval(expected)
                for moment,budget in ((0,20000),(3,40000),(6,20000)):
                    now[0]=moment
                    if moment==6:memory[0]=(7500000,500000,10000000)
                    result=model(tokens);mx.eval(result)
                    self.assertTrue(mx.allclose(expected,result,atol=2e-4,rtol=2e-4).item())
                    self.assertEqual(store.cache.budget,budget)
                    self.assertLessEqual(store.cache.bytes,budget)
                store.close()
                self.assertFalse(controller.snapshot()['models'])

    def test_batched_rows_duplicate_routes_and_eviction_match(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            config = tiny_config('qwen3_moe')
            reference = create_checkpoint(directory, config)
            streamed, _, store = load_streamed_model(directory, cache_bytes=40000, chunk_rows=2)
            x = mx.random.normal((2,3,128))
            ids = mx.array([[[0,1],[2,3],[0,0]], [[1,2],[3,0],[2,1]]])
            a = reference.layers[0].mlp.switch_mlp(x,ids)
            b = streamed.layers[0].mlp.switch_mlp(x,ids)
            mx.eval(a,b)
            self.assertTrue(mx.allclose(a,b,atol=2e-4,rtol=2e-4).item())
            self.assertGreater(store.cache.misses,0)
            self.assertLessEqual(store.cache.bytes,40000)
            store.close()

    def test_cache_reuses_experts_across_calls_and_fds_close_with_model(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            create_checkpoint(directory,tiny_config('qwen3_moe'))
            model, _, store = load_streamed_model(directory,cache_bytes=1024**2)
            x,ids = mx.ones((1,1,128)),mx.array([[[0,1]]])
            mx.eval(model.layers[0].mlp.switch_mlp(x,ids))
            read = store.files.bytes_read
            mx.eval(model.layers[0].mlp.switch_mlp(x,ids))
            self.assertEqual(store.files.bytes_read,read)
            self.assertGreater(store.cache.hits,0)
            del model
            gc.collect()
            self.assertTrue(store.closed)
            self.assertFalse(store.files.descriptors)

    def test_bfloat16_file_payload_preserves_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            values=mx.array([[1.5,-2.25,0.125]],dtype=mx.bfloat16)
            mx.save_safetensors(str(directory/'model.safetensors'),{'sample':values})
            with TensorFiles(directory) as files:
                actual=tensor_array(files.read('sample'))
                self.assertEqual(actual.dtype,mx.bfloat16)
                self.assertEqual(actual.tolist(),values.tolist())

    def test_server_loader_hook_streams_allowlisted_models_and_never_falls_back_on_failure(self):
        from expert_streaming_mlx import install_loader
        from omlx.utils import model_loading
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            create_checkpoint(directory,tiny_config('qwen3_moe'))
            original=model_loading.lm_load_compat
            try:
                with patch.object(model_loading,'lm_load_compat',return_value=('normal','tokenizer')) as fallback:
                    from adaptive_expert_cache import AdaptiveExpertCaches
                    adaptive=AdaptiveExpertCaches(100000,8000000,lambda:(1000000,8000000,10000000))
                    install_loader({str(directory):{'cache_bytes':40000,'chunk_rows':2,
                        'adaptive':{'min_bytes':10000,'max_bytes':80000}}},adaptive)
                    with patch('expert_streaming_mlx.load_tokenizer',return_value='local-tokenizer'):
                        model,tokenizer=model_loading.lm_load_compat(str(directory),tokenizer_config={})
                        self.assertEqual(tokenizer,'local-tokenizer')
                        self.assertIsInstance(model.layers[0].mlp.switch_mlp,StreamedSwitchGLU)
                        self.assertIs(model.layers[0].mlp.switch_mlp.store.adaptive,adaptive)
                        self.assertEqual(adaptive.snapshot()['allocated_budget_bytes'],40000)
                        self.assertEqual(model_loading.lm_load_compat('/not/allowlisted'),('normal','tokenizer'))
                        self.assertEqual(fallback.call_count,1)
                        (directory/'model.safetensors').write_bytes(b'broken')
                        with self.assertRaises(ValueError): model_loading.lm_load_compat(str(directory))
                        self.assertEqual(fallback.call_count,1)
            finally:
                model_loading.lm_load_compat=original

    def test_two_models_work_on_separate_engine_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        from omlx.utils.model_loading import materialize_lazy_state
        with tempfile.TemporaryDirectory() as folder:
            models=[]
            for kind in ('qwen3_moe','glm4_moe_lite'):
                directory=Path(folder)/kind;directory.mkdir()
                create_checkpoint(directory,tiny_config(kind))
                model,_,store=load_streamed_model(directory,cache_bytes=40000,chunk_rows=2)
                materialize_lazy_state(model)
                models.append((model,store))
            from adaptive_expert_cache import AdaptiveExpertCaches
            adaptive=AdaptiveExpertCaches(100000,8000000,lambda:(7500000,500000,10000000), interval=.0001)
            for i,(_,store) in enumerate(models):
                adaptive.register(store,str(i),10000,80000)
                store.adaptive=adaptive
            def generate(item):
                model,_=item
                stream=mx.new_thread_local_stream(mx.default_device())
                with mx.stream(stream):
                    cache=make_prompt_cache(model)
                    for token in (1,2,3):
                        logits=model(mx.array([[token]]),cache=cache)
                        mx.eval(logits)
                        self.assertTrue(mx.all(mx.isfinite(logits)).item())
                return True
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(list(executor.map(generate,models)),[True,True])
            for _,store in models: store.close()
