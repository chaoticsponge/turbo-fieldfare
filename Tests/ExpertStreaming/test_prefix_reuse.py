"""Exercise the pinned engine's cache implementation with tiny synthetic data."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from test_streamed_moe import create_checkpoint, tiny_config


class SharedPrefixTests(unittest.TestCase):
    def test_concurrent_agents_share_only_identical_model_and_token_prefixes(self):
        manager=PagedCacheManager(block_size=4,max_blocks=16,initial_blocks=4,model_name='coder')
        tokens=list(range(8));blocks=manager.get_new_blocks(2)
        manager.cache_full_blocks(blocks,tokens,0,2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(manager.get_computed_blocks,[tokens+[20],tokens[:4]+[99,98,97,96]]))
        self.assertEqual([count for _,count in results],[8,4])
        self.assertEqual(results[0][0][0].block_id,results[1][0][0].block_id)
        self.assertEqual(manager.get_computed_blocks([99]+tokens[1:])[1],0)
        key=compute_block_hash(None,tokens[:4],model_name='coder')
        self.assertNotEqual(key,compute_block_hash(None,tokens[:4],model_name='research'))
        self.assertNotEqual(key,compute_block_hash(None,tokens[:4],model_name='coder',extra_keys=('image-a',)))

    def test_ssd_prefix_survives_restart_and_preserves_continuation_outputs(self):
        for kind in ('qwen3_moe','glm4_moe_lite'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);model=create_checkpoint(root,tiny_config(kind))
                cache=make_prompt_cache(model);tokens=[1,2,3,4]
                logits=model(mx.array([tokens]),cache=cache);mx.eval(logits)
                states=[entry.state for entry in cache];mx.eval(states)
                block_hash=compute_block_hash(None,tokens,model_name=kind)
                def open_store():
                    return PagedSSDCacheManager(cache_dir=root/'cache',max_size_bytes=8*1024**2,
                        hot_cache_max_bytes=0,expected_model_name=kind,expected_num_layers=len(states),
                        expected_block_size=4,expected_block_size_tokens=4)
                store=open_store()
                try:
                    self.assertTrue(store.save_block(block_hash,states,token_count=4,model_name=kind,
                        layer_cache_types=['KVCache']*len(states),layer_meta_states=[()]*len(states)))
                finally:store.close()
                reopened=open_store()
                try:
                    loaded=reopened.load_block(block_hash)
                    self.assertIsNotNone(loaded)
                    restored=make_prompt_cache(model)
                    for entry,state in zip(restored,loaded):entry.state=state
                    expected=model(mx.array([[5]]),cache=cache)
                    actual=model(mx.array([[5]]),cache=restored)
                    mx.eval(expected,actual)
                    self.assertTrue(mx.allclose(expected,actual,atol=2e-4,rtol=2e-4).item())
                    manager=PagedCacheManager(block_size=4,max_blocks=16,initial_blocks=4,model_name=kind)
                    manager.set_paged_ssd_cache_manager(reopened)
                    self.assertEqual(manager.get_computed_blocks(tokens+[9])[1],4)
                    self.assertIsNone(reopened.load_block(compute_block_hash(None,tokens,model_name='different-model')))
                finally:reopened.close()
