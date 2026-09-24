import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from adaptive_expert_cache import AdaptiveExpertCaches
from expert_streaming import ExpertCache, LayerExpertCache
from test_launcher import launcher


class Store:
    def __init__(self, budget=100):self.cache=LayerExpertCache(budget,['a','b'])


class AdaptiveCacheTests(unittest.TestCase):
    def setUp(self):
        self.now=0
        self.memory=(100,800,1000)
        self.controller=AdaptiveExpertCaches(300,800,lambda:self.memory,
                    clock=lambda:self.now,interval=2,step_bytes=50)

    def register(self,name='model',budget=100):
        store=Store(budget)
        self.controller.register(store,name,20,200)
        return store

    def misses(self,store):
        for i in range(40):store.cache.get(('a',i),30,lambda:bytes(30))

    def test_growth_pool_limit_and_per_model_limit(self):
        a=self.register('a');b=self.register('b')
        for moment in (3,6,9):
            self.now=moment
            for store in (a,b):
                self.misses(store);self.controller.boundary(store)
            self.assertLessEqual(sum(s.cache.budget for s in (a,b)),300)
        self.assertEqual(a.cache.budget,150)
        self.assertEqual(b.cache.budget,150)
        self.controller.unregister(b)
        self.now=12;self.misses(a);self.controller.boundary(a)
        self.assertEqual(a.cache.budget,200)
        self.now=15;self.misses(a);self.controller.boundary(a)
        self.assertEqual(a.cache.budget,200)

    def test_pressure_shrinks_and_really_evicts_to_floor(self):
        store=self.register()
        for layer in ('a','b'):
            for i in range(5):store.cache.get((layer,i),10,lambda:bytes(10))
        self.memory=(750,50,1000)
        for time,budget in ((3,50),(6,25),(9,20),(12,20)):
            self.now=time;self.controller.boundary(store)
            self.assertEqual(store.cache.budget,budget)
            self.assertLessEqual(store.cache.bytes,budget)
        self.assertGreater(store.cache.evictions,0)
        self.assertEqual(self.controller.snapshot()['pressure'],'high')

    def test_cooldown_and_neutral_pressure_prevent_oscillation(self):
        store=self.register();self.misses(store)
        self.now=1;self.controller.boundary(store)
        self.assertEqual(store.cache.budget,100)
        self.memory=(600,150,1000);self.now=3
        self.controller.boundary(store)
        self.assertEqual(store.cache.budget,100)
        self.assertEqual(self.controller.snapshot()['pressure'],'hold')

    def test_telemetry_failure_shrinks_and_weak_registry_does_not_keep_models_alive(self):
        store=self.register()
        def broken():raise OSError('unavailable')
        self.controller.sample=broken
        self.now=3;self.controller.boundary(store)
        self.assertEqual(store.cache.budget,50)
        self.assertEqual(self.controller.snapshot()['pressure'],'unavailable')
        del store;gc.collect()
        self.assertEqual(self.controller.snapshot()['allocated_budget_bytes'],0)

    def test_growth_headroom_is_shared_between_models_between_samples(self):
        a=self.register('a');b=self.register('b')
        # 800*.70 - 508 permits less than 50 bytes growth after the margin.
        self.memory=(508,800,1000);self.now=3
        for store in (a,b):self.misses(store);self.controller.boundary(store)
        self.assertLessEqual(a.cache.budget+b.cache.budget,249)

    def test_resize_retains_popular_experts_and_layer_caps(self):
        cache=ExpertCache(30,'lfu')
        for i in range(3):cache.get(i,10,lambda:bytes(10))
        for _ in range(5):cache.get(2,10,lambda:self.fail('Unexpected miss'))
        cache.resize(10)
        self.assertEqual(list(cache.entries),[2])
        cache.resize(40)
        self.assertEqual(cache.bytes,10)
        cache.resize(0)
        self.assertEqual(cache.bytes,0)
        with self.assertRaises(ValueError):cache.resize(-1)
        layers=LayerExpertCache(101,['a','b','c']);layers.resize(20)
        self.assertEqual(sum(c.budget for c in layers.layers.values()),20)

    def test_unloaded_models_keep_startup_space_and_can_reload_after_shrinking(self):
        controller=AdaptiveExpertCaches(300,800,lambda:self.memory,clock=lambda:self.now,
            interval=2,step_bytes=50,initial_budgets={'a':100,'b':100})
        a=Store();controller.register(a,'a',20,300)
        for moment in (3,6,9):
            self.now=moment;self.misses(a);controller.boundary(a)
        self.assertEqual(a.cache.budget,200)
        b=Store();controller.register(b,'b',20,300)
        self.memory=(750,50,1000);self.now=12;controller.boundary(b)
        self.assertEqual(b.cache.budget,50)
        controller.unregister(b)
        b=Store();controller.register(b,'b',20,300)
        self.assertEqual(b.cache.budget,100)
        self.assertEqual(controller.snapshot()['committed_pool_bytes'],300)

    def test_configuration_reserves_peak_cache_and_cli_rejects_impossible_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder)
            config=launcher.settings(state,2,65536,48)
            limits={'min_bytes':64*1024**2,'max_bytes':1024*1024**2,'pool_bytes':1024*1024**2}
            profile={'planned_weight_bytes':1500000000,'cache_bytes':256*1024**2}
            launcher.configure_fleet(state,['coder'],state,config,launcher.command(state,state,8080,2,48),
                2,65536,stream_profiles={'coder':profile},adaptive_cache=limits)
            saved=json.loads((state/'routes.json').read_text())
            entry=saved['routes']['coder']
            self.assertEqual(entry['resident_weight_bytes'],1500000000+768*1024**2)
            self.assertEqual(entry['expert_streaming']['adaptive'],limits)
            self.assertEqual(saved['adaptive_expert_cache']['memory_ceiling'],48*1024**3)
        for args in [['--adaptive-expert-cache'],
                     ['--fleet','--stream-experts','--adaptive-expert-cache','--expert-cache-mb','512','--expert-cache-max-mb','256'],
                     ['--fleet','--stream-experts','--adaptive-expert-cache','--expert-cache-pool-mb','256']]:
            result=subprocess.run([sys.executable,str(launcher.ROOT/'Scripts/serve-qwen-agents.py'),*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
