import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from agent_models import CONTEXT_DEFAULTS, ROLES, context_budgets, route_config
from agent_routing import AgentRouter, OUTPUT_RESERVATION, RouteError, resolve
from agent_runtime import install_context_validation
from test_launcher import launcher
import test_routing
from test_routing import prompt


class ContextBudgetTests(unittest.TestCase):
    def test_defaults_ceiling_and_explicit_overrides(self):
        self.assertEqual(context_budgets(ROLES), CONTEXT_DEFAULTS)
        limits = context_budgets(ROLES, 16384, {'coder': 8192})
        self.assertEqual(limits['research'], 16384)
        self.assertEqual(limits['extract'], 8192)
        self.assertEqual(limits['coder'], 8192)
        self.assertEqual(context_budgets(ROLES, overrides={'worker': 32768})['worker'], 32768)
        for roles, cap, overrides in [(['coder'],65536,{'worker':8192}),
                                      (ROLES,8192,{'coder':16384}),
                                      (ROLES,65536,{'coder':123})]:
            with self.assertRaises(ValueError): context_budgets(roles, cap, overrides)

    def test_launcher_rejects_invalid_overrides_before_preflight(self):
        for args in [['--role-context','worker=8192'],
                     ['--fleet','--role-context','invalid=8192'],
                     ['--fleet','--role-context','coder=bad'],
                     ['--fleet','--role-context','coder=8192','--role-context','coder=16384'],
                     ['--fleet','--context','8192','--role-context','coder=32768']]:
            result = subprocess.run([sys.executable, str(launcher.ROOT/'Scripts/serve-qwen-agents.py'), *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('--role-context', result.stderr)

    def test_written_routes_and_engine_settings_agree(self):
        from omlx.model_settings import ModelSettingsManager
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder)
            config=launcher.settings(state,2,32768,48)
            launcher.configure_fleet(state,list(ROLES),state,config,
                launcher.command(state,state,8080,2,48),2,32768,
                role_contexts={'worker':8192})
            routes=json.loads((state/'routes.json').read_text())['routes']
            manager=ModelSettingsManager(state)
            for role, limit in context_budgets(ROLES,32768,{'worker':8192}).items():
                self.assertEqual(routes[role]['max_context_window'],limit)
                self.assertEqual(manager.get_settings(routes[role]['model']).max_context_window,limit)

    def test_output_aliases_defaults_and_invalid_budgets(self):
        routes=route_config(ROLES)
        for role,entry in routes.items(): entry['max_context_window']=CONTEXT_DEFAULTS[role]
        for model in ('extract',routes['extract']['model'],'auto'):
            request=prompt('Extract emails',model)
            original=copy.deepcopy(request['messages'])
            result=resolve('/v1/chat/completions',request,routes)[1]
            self.assertEqual(result['max_tokens'],2048)
            self.assertEqual(result['messages'],original)
        for key in ('max_tokens','max_completion_tokens'):
            request=prompt('hello','worker');request[key]=1000
            self.assertEqual(resolve('/v1/chat/completions',request,routes)[1]['max_tokens'],1000)
        for invalid in (0,-1,True,'100',1.5,8192):
            request=prompt('Extract emails');request['max_tokens']=invalid
            with self.assertRaises(RouteError): resolve('/v1/chat/completions',request,routes)
        request=prompt('hello');request.update(max_tokens=100,max_completion_tokens=200)
        with self.assertRaises(RouteError): resolve('/v1/chat/completions',request,routes)

    def test_engine_token_validation_reserves_output_and_preserves_prompt_checks(self):
        from fastapi import HTTPException
        seen=[]
        def original(tokens, model):
            seen.append((tokens,model))
            if tokens>8192: raise HTTPException(400,'Prompt too long')
        server=SimpleNamespace(validate_context_window=original,get_max_context_window=lambda model:8192)
        install_context_validation(server)
        token=OUTPUT_RESERVATION.set(2048)
        try:
            server.validate_context_window(6144,'extract')
            with self.assertRaises(HTTPException) as failure: server.validate_context_window(6145,'extract')
            self.assertEqual(failure.exception.status_code,400)
            self.assertIn('reserved output',failure.exception.detail)
            with self.assertRaises(HTTPException) as failure: server.validate_context_window(8193,'extract')
            self.assertEqual(failure.exception.detail,'Prompt too long')
        finally: OUTPUT_RESERVATION.reset(token)
        server.validate_context_window(8192,'extract')
        self.assertEqual(seen[-1],(8192,'extract'))


class ConcurrentContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_reservations_are_isolated_and_discovery_reports_limits(self):
        routes=route_config(ROLES)
        for role,entry in routes.items(): entry['max_context_window']=CONTEXT_DEFAULTS[role]
        ready=asyncio.Event(); observed=[]
        async def backend(scope,receive,send):
            body=json.loads((await receive())['body'])
            observed.append((body['max_tokens'],OUTPUT_RESERVATION.get()))
            if len(observed)==2: ready.set()
            await asyncio.wait_for(ready.wait(),1)
            self.assertEqual(OUTPUT_RESERVATION.get(),body['max_tokens'])
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        async def switch(model,protected,active): pass
        router=AgentRouter(backend,routes,switch)
        caller=test_routing.MiddlewareTests()
        requests=[dict(prompt('Debug code'),max_tokens=100),dict(prompt('Research evidence'),max_completion_tokens=200)]
        await asyncio.gather(*(caller.call(router,r) for r in requests))
        self.assertEqual(sorted(observed),[(100,100),(200,200)])
        self.assertEqual(OUTPUT_RESERVATION.get(),0)
        health=await caller.call(router,path='/health',method='GET')
        self.assertEqual(json.loads(health[-1]['body'])['context_budgets'],CONTEXT_DEFAULTS)
        models=await caller.call(router,path='/v1/models',method='GET')
        items={item['id']:item for item in json.loads(models[-1]['body'])['data']}
        self.assertEqual(items['extract']['max_context_window'],8192)
        self.assertNotIn('max_context_window',items['auto'])
