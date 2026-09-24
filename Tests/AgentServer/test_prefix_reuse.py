import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from agent_prefix_cache import prefix_cache_size, prefix_namespace, prefix_cache_snapshot
from agent_models import ROLES, route_config
from test_launcher import launcher


class PrefixReuseTests(unittest.TestCase):
    def test_fleet_default_and_explicit_disable(self):
        self.assertEqual(prefix_cache_size(True,None),4)
        self.assertEqual(prefix_cache_size(False,None),0)
        self.assertEqual(prefix_cache_size(True,0),0)
        self.assertEqual(prefix_cache_size(True,8),8)

    def test_namespace_stability_and_compatibility_boundaries(self):
        a=(Path('/models/a'),{'repoID':'repo/a','revision':'a'*40})
        b=(Path('/models/b'),{'repoID':'repo/b','revision':'b'*40})
        runtime={'versions':{'mlx':'1','omlx':'2'}}
        base=prefix_namespace([a,b],runtime)
        self.assertEqual(base,prefix_namespace([b,a],runtime))
        self.assertNotEqual(base,prefix_namespace([a,b],{'versions':{'mlx':'new'}}))
        self.assertNotEqual(base,prefix_namespace([a,b],runtime,['a']))
        changed=(b[0],{**b[1],'revision':'c'*40})
        self.assertNotEqual(base,prefix_namespace([a,changed],runtime))

    def test_fleet_persists_cache_configuration_and_keeps_hot_cache_off(self):
        from omlx.settings import GlobalSettings
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder);cache=state/'prefix-cache'/'revision'
            settings=launcher.settings(state,2,65536,48,4,cache)
            command=launcher.configure_fleet(state,list(ROLES),state,settings,
                launcher.command(state,state,8080,2,48,4,cache),2,65536)
            (state/'settings.json').write_text(json.dumps(settings))
            parsed=GlobalSettings.load(state)
            self.assertTrue(parsed.cache.enabled)
            self.assertEqual(parsed.cache.hot_cache_max_size,'0')
            self.assertIn('--paged-ssd-cache-dir',command)
            self.assertNotIn('--no-cache',command)
            reported=json.loads((state/'routes.json').read_text())['prefix_cache']
            self.assertEqual(reported['namespace'],'revision')
            self.assertTrue(reported['enabled'])
            self.assertEqual(reported['hot_cache_bytes'],0)

    def test_statistics_distinguish_unloaded_unavailable_and_ready(self):
        routes=route_config(ROLES)
        class Pool:
            def get_entry(self,model):
                if model==routes['worker']['model']:return SimpleNamespace(engine=None)
                if model==routes['coder']['model']:
                    return SimpleNamespace(engine=SimpleNamespace(get_cache_stats=lambda:SimpleNamespace(hits=2,misses=1,tokens_saved=512)))
                return SimpleNamespace(engine=SimpleNamespace(get_cache_stats=lambda:None))
        snapshot=prefix_cache_snapshot(Pool(),routes,{'enabled':True})
        self.assertEqual(snapshot['models']['worker']['state'],'unloaded')
        self.assertEqual(snapshot['models']['coder']['tokens_saved'],512)
        self.assertEqual(snapshot['models']['research']['state'],'unavailable')
        self.assertNotIn('embed',snapshot['models'])
        self.assertEqual(prefix_cache_snapshot(None,routes,{'enabled':False})['models'],{})
