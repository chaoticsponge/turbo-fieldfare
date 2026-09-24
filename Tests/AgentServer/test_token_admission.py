import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent_admission import GIB, cache_profile, estimate_session
from agent_models import route_config
from agent_routing import AgentRouter, ModelGate, OUTPUT_RESERVATION, TOKEN_RESERVATION, RouteError, resolve
from agent_runtime import install_context_validation, reclaim_idle
from test_launcher import launcher
import test_routing


def coder_config():
    return {'model_type':'qwen3_moe','num_hidden_layers':48,'num_key_value_heads':4,
            'head_dim':128,'hidden_size':2048,'torch_dtype':'bfloat16'}


def route():
    return {**route_config(['coder'])['coder'], 'max_context_window':32768,
            'cache_profile':cache_profile(coder_config())}


def estimate(text='hello', output=100):
    body={'messages':[{'role':'user','content':text}], 'max_tokens':output}
    return estimate_session('/v1/chat/completions',body,route())


class EstimateTests(unittest.TestCase):
    def test_cache_layouts_match_gqa_hybrid_and_latent_arrays(self):
        self.assertEqual(cache_profile(coder_config())['kv_bytes_per_token'],98304)
        glm={'model_type':'glm4_moe_lite','num_hidden_layers':47,'hidden_size':2048,
             'kv_lora_rank':512,'qk_rope_head_dim':64,'num_attention_heads':20,'dtype':'bfloat16'}
        profile=cache_profile(glm)
        self.assertEqual(profile['kv_bytes_per_token'],54144)
        self.assertEqual(profile['attention_workspace_bytes_per_pair'],80)
        hybrid={**coder_config(),'model_type':'qwen3_5_text','num_hidden_layers':64,
                'full_attention_interval':4,'head_dim':256,'linear_num_value_heads':48,
                'linear_num_key_heads':16,'linear_key_head_dim':128,'linear_value_head_dim':128,
                'linear_conv_kernel_dim':4}
        profile=cache_profile({'text_config':hybrid})
        self.assertEqual(profile['kv_bytes_per_token'],65536)
        self.assertEqual(profile['state_bytes'],48*(48*128*128*4+3*(2*16*128+48*128)*2))
        self.assertEqual(cache_profile({'model_type':'unknown'}),{'layout':'fallback'})
        self.assertEqual(cache_profile({**coder_config(),'num_hidden_layers':0}),{'layout':'fallback'})

    def test_short_requests_save_headroom_and_long_context_can_exceed_old_reservation(self):
        small=estimate()
        self.assertLess(small.session_bytes,3*GIB)
        self.assertGreater(estimate('x'*30000,2000).session_bytes,3*GIB)
        self.assertGreater(estimate(output=4000).session_bytes,small.session_bytes)
        self.assertGreater(estimate('x'*4000).session_bytes,small.session_bytes)
        self.assertEqual(estimate('x'*100000).total_tokens,32768)

    def test_unicode_tools_and_history_count_images_reserve_context_ceiling(self):
        cfg=route()
        body={'messages':[{'role':'user','content':'hello'}],'max_tokens':100}
        base=estimate_session('/v1/chat/completions',body,cfg)
        body['tools']=[{'type':'function','function':{'name':'tool','description':'工具'*1000}}]
        tools=estimate_session('/v1/chat/completions',body,cfg)
        self.assertGreater(tools.prompt_tokens,base.prompt_tokens)
        body['messages'].append({'role':'tool','content':'x'*1000})
        self.assertGreater(estimate_session('/v1/chat/completions',body,cfg).prompt_tokens,tools.prompt_tokens)
        body['messages']=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'x'}}]}]
        image=estimate_session('/v1/chat/completions',body,cfg)
        self.assertEqual(image.total_tokens,32768)
        self.assertEqual(image.method,'image-context-ceiling')
        self.assertGreater(image.session_bytes,estimate('x'*100000).session_bytes)

    def test_unknown_and_retrieval_keep_fixed_reservations(self):
        for path,cfg in [('/v1/chat/completions',{}),('/v1/embeddings',route()),('/v1/rerank',route())]:
            result=estimate_session(path,{},cfg)
            self.assertEqual(result.session_bytes,3*GIB)
            self.assertIsNone(result.total_tokens)

    def test_launcher_reads_local_config_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            state=Path(folder)
            model=state/'Qwen3-Coder-30B-A3B-Instruct-4bit'
            model.mkdir();(model/'config.json').write_text(json.dumps(coder_config()))
            config=launcher.settings(model,2,32768,48)
            launcher.configure_fleet(state,['coder'],state,config,
                launcher.command(model,state,8080,2,48),2,32768)
            result=json.loads((state/'routes.json').read_text())['routes']['coder']
            self.assertEqual(result['cache_profile']['kv_bytes_per_token'],98304)

    def test_actual_token_count_cannot_outgrow_admitted_estimate(self):
        from fastapi import HTTPException
        server=SimpleNamespace(validate_context_window=lambda *args:None,
                               get_max_context_window=lambda model:32768)
        install_context_validation(server)
        output=OUTPUT_RESERVATION.set(100);total=TOKEN_RESERVATION.set(2000)
        try:
            server.validate_context_window(1900,'coder')
            with self.assertRaises(HTTPException) as failure:server.validate_context_window(1901,'coder')
            self.assertEqual(failure.exception.status_code,400)
            self.assertIn('admission token estimate',failure.exception.detail)
        finally:
            OUTPUT_RESERVATION.reset(output);TOKEN_RESERVATION.reset(total)


class VariableGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_weights_variable_sessions_and_out_of_order_release(self):
        projected=[]
        async def switch(model,protected,total):projected.append(total)
        gate=ModelGate(switch,3,weights={'a':10},budget_bytes=30)
        small=await gate.acquire('a',2)
        large=await gate.acquire('a',12)
        self.assertEqual(gate.active_session_bytes,14)
        self.assertEqual(projected,[2,14])
        self.assertTrue(gate.fits('a',6));self.assertFalse(gate.fits('a',7))
        await gate.release(large)
        self.assertEqual(gate.active_session_bytes,2)
        await gate.release(small)
        self.assertEqual(gate.active_session_bytes,0)
        self.assertFalse(gate.leases)

    async def test_large_request_queues_and_cancelled_waiter_reserves_nothing(self):
        async def switch(*args):pass
        gate=ModelGate(switch,2,weights={'a':10,'b':10},budget_bytes=30)
        lease=await gate.acquire('a',3)
        waiter=asyncio.create_task(gate.acquire('b',8))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        self.assertEqual(gate.active_session_bytes,3)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):await waiter
        with self.assertRaises(RouteError):await gate.acquire('b',21)
        await gate.release(lease)
        self.assertEqual(gate.active_session_bytes,0)

    async def test_idle_eviction_uses_actual_sum_not_request_count(self):
        class Pool:
            loaded={'a','idle'}
            def get_loaded_model_ids(self):return list(self.loaded)
            def get_entry(self,model):return SimpleNamespace(last_access=0)
            async def unload_if_idle_unpinned(self,model):self.loaded.remove(model);return True
        pool=Pool()
        await reclaim_idle(pool,{'a'},15,{'a':10,'idle':10},30)
        self.assertEqual(pool.loaded,{'a'})

    async def test_stream_holds_reservation_and_failure_releases_it(self):
        started=asyncio.Event();finish=asyncio.Event()
        routes={'coder':route()}
        request={**test_routing.prompt('Debug repository'),'max_tokens':100}
        expected=estimate_session('/v1/chat/completions',resolve('/v1/chat/completions',dict(request),routes)[1],routes['coder'])
        async def backend(scope,receive,send):
            self.assertEqual(TOKEN_RESERVATION.get(),expected.total_tokens)
            await send({'type':'http.response.start','status':200,'headers':[]})
            started.set();await finish.wait()
            await send({'type':'http.response.body','body':b'data: [DONE]\n\n'})
        async def switch(*args):pass
        router=AgentRouter(backend,routes,switch)
        caller=test_routing.MiddlewareTests()
        task=asyncio.create_task(caller.call(router,request))
        await asyncio.wait_for(started.wait(),1)
        self.assertEqual(router.gate.active_session_bytes,expected.session_bytes)
        finish.set();sent=await task
        self.assertEqual(dict(sent[0]['headers'])[b'x-agent-session-bytes'],str(expected.session_bytes).encode())
        self.assertEqual(router.gate.active_session_bytes,0)
        async def broken(*args):raise RuntimeError('failed')
        router.app=broken
        with self.assertRaises(RuntimeError):await caller.call(router,request)
        self.assertEqual(router.gate.active_session_bytes,0)
        self.assertIsNone(TOKEN_RESERVATION.get())
