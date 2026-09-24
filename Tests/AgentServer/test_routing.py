import asyncio
import copy
import json
import unittest

from agent_models import ROLES, route_config
from agent_routing import AgentRouter, ModelGate, RouteError, resolve


def prompt(text, model='auto'):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}]}


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.routes = route_config(ROLES)

    def test_specialists_and_general_fallback(self):
        cases = {'Write a Python script to scrape jobs': 'coder',
                 'Debug the repository build': 'coder',
                 'Research remote job markets and cite evidence': 'research',
                 'Extract email addresses from this HTML': 'extract',
                 'Classify these job descriptions': 'extract',
                 'Draft a friendly email': 'worker',
                 'hello': 'worker', 'x' * 24001: 'research'}
        for text, expected in cases.items():
            with self.subTest(text=text[:60]):
                self.assertEqual(resolve('/v1/chat/completions', prompt(text), self.routes)[0], expected)

    def test_tool_loop_stays_on_user_intent_and_preserves_tools(self):
        body = prompt('Fix a bug in the repository')
        body.update({'stream': True, 'tools': [{'type': 'function', 'function': {'name': 'read'}}]})
        body['messages'] += [{'role': 'assistant', 'tool_calls': [{'id': 'abc', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
                             {'role': 'tool', 'tool_call_id': 'abc', 'content': 'research extract classify ' * 3000}]
        original = copy.deepcopy(body)
        role, routed = resolve('/v1/chat/completions', body, self.routes)
        self.assertEqual(role, 'coder')
        self.assertEqual(routed['messages'], original['messages'])
        self.assertEqual(routed['tools'], original['tools'])
        self.assertTrue(routed['stream'])

    def test_explicit_role_and_sampling_override(self):
        body = prompt('Write some code', 'research')
        body['temperature'] = 0.2
        role, routed = resolve('/v1/chat/completions', body, self.routes)
        self.assertEqual(role, 'research')
        self.assertEqual(routed['temperature'], 0.2)
        self.assertEqual(resolve('/v1/chat/completions', prompt('hello', self.routes['coder']['model']), self.routes)[0], 'coder')

    def test_images_never_reach_text_only_specialists(self):
        body = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'debug'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,a'}}]}]}
        self.assertEqual(resolve('/v1/chat/completions', copy.deepcopy(body), self.routes)[0], 'worker')
        body['model'] = 'coder'
        with self.assertRaisesRegex(RouteError, 'images'):
            resolve('/v1/chat/completions', body, self.routes)

    def test_missing_specialist_fails_without_silent_fallback(self):
        with self.assertRaises(RouteError) as failure:
            resolve('/v1/chat/completions', prompt('Debug this repository'), {'worker': self.routes['worker']})
        self.assertEqual(failure.exception.status, 503)

    def test_embedding_and_reranking_are_separate_bounded_endpoints(self):
        self.assertEqual(resolve('/v1/embeddings', {'input': ['a', 'b']}, self.routes)[0], 'embed')
        self.assertEqual(resolve('/v1/rerank', {'query': 'job', 'documents': ['one', 'two']}, self.routes)[0], 'rerank')
        for path, body in [('/v1/chat/completions', prompt('hi', 'embed')),
                           ('/v1/embeddings', {'model': 'worker', 'input': 'a'}),
                           ('/v1/embeddings', {'input': ['a'] * 33}),
                           ('/v1/rerank', {'query': 'q', 'documents': ['x' * 16385]})]:
            with self.assertRaises(RouteError):
                resolve(path, body, self.routes)


class GateTests(unittest.IsolatedAsyncioTestCase):
    async def test_different_models_overlap_when_weights_and_sessions_fit(self):
        admitted = []
        async def switch(model, protected, active): admitted.append((model, protected, active))
        gate = ModelGate(switch, 2, weights={'coder': 16, 'research': 16},
                         budget_bytes=40, session_bytes=3)
        await gate.acquire('coder')
        await asyncio.wait_for(gate.acquire('research'), 1)
        self.assertEqual(gate.active, 2)
        self.assertEqual(admitted[-1], ('research', {'coder', 'research'}, 6))
        await gate.release('coder')
        await gate.release('research')

    async def test_oversized_pair_queues_fairly_but_shared_weights_count_once(self):
        async def switch(model, protected, active): pass
        gate = ModelGate(switch, 3, weights={'worker': 28, 'coder': 17},
                         budget_bytes=41, session_bytes=3)
        await gate.acquire('worker')
        await gate.acquire('worker')
        coder = asyncio.create_task(gate.acquire('coder'))
        await asyncio.sleep(0)
        later_worker = asyncio.create_task(gate.acquire('worker'))
        await asyncio.sleep(0)
        await gate.release('worker')
        self.assertFalse(coder.done())
        self.assertFalse(later_worker.done())
        await gate.release('worker')
        await asyncio.wait_for(coder, 1)
        self.assertFalse(later_worker.done())
        await gate.release('coder')
        await asyncio.wait_for(later_worker, 1)
        await gate.release('worker')
        self.assertEqual(gate.active, 0)

    async def test_cancelled_waiter_does_not_block_queue(self):
        async def switch(model, protected, active): pass
        gate = ModelGate(switch, 1, max_waiters=1)
        await gate.acquire('worker')
        pending = asyncio.create_task(gate.acquire('coder'))
        await asyncio.sleep(0)
        with self.assertRaises(RouteError) as error:
            await gate.acquire('research')
        self.assertEqual(error.exception.status, 429)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError): await pending
        await gate.release('worker')
        await asyncio.wait_for(gate.acquire('research'), 1)
        await gate.release('research')
        self.assertFalse(gate.waiters)

    async def test_failed_unload_does_not_admit_next_model(self):
        async def switch(model, protected, active): raise RouteError('still busy', 503)
        gate = ModelGate(switch)
        with self.assertRaises(RouteError): await gate.acquire('coder')
        self.assertEqual(gate.active, 0)
        self.assertFalse(gate.waiters)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, router, body=None, path='/v1/chat/completions', method='POST', gone=None):
        sent, delivered = [], False
        gone = gone or asyncio.Event()
        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': json.dumps(body).encode()}
            await gone.wait()
            return {'type': 'http.disconnect'}
        async def send(message): sent.append(message)
        await router({'type': 'http', 'path': path, 'method': method, 'headers': [(b'host', b'127.0.0.1:8080'), (b'content-type', b'application/json')]}, receive, send)
        return sent

    async def test_stream_passthrough_and_lease_until_final_body(self):
        started, finish = asyncio.Event(), asyncio.Event()
        received = []
        async def backend(scope, receive, send):
            received.append(json.loads((await receive())['body']))
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'data: tool_delta\n\n', 'more_body': True})
            started.set()
            await finish.wait()
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        async def switch(model, protected, active): pass
        router = AgentRouter(backend, route_config(ROLES), switch)
        task = asyncio.create_task(self.call(router, prompt('Debug repository')))
        await asyncio.wait_for(started.wait(), 1)
        self.assertEqual(router.gate.active, 1)
        finish.set()
        sent = await task
        self.assertEqual(router.gate.active, 0)
        self.assertIn((b'x-agent-role', b'coder'), sent[0]['headers'])
        self.assertEqual(sent[1]['body'], b'data: tool_delta\n\n')
        self.assertEqual(sent[-1]['body'], b'data: [DONE]\n\n')
        self.assertEqual(received[0]['model'], ROLES['coder']['directory'])

    async def test_queued_disconnect_releases_waiter(self):
        async def backend(scope, receive, send): self.fail('Disconnected request reached inference')
        async def switch(model, protected, active): pass
        router = AgentRouter(backend, route_config(ROLES), switch, concurrency=1)
        await router.gate.acquire('busy')
        gone = asyncio.Event()
        task = asyncio.create_task(self.call(router, prompt('hello'), gone=gone))
        for _ in range(5): await asyncio.sleep(0)
        gone.set()
        await asyncio.wait_for(task, 1)
        self.assertFalse(router.gate.waiters)
        self.assertEqual(router.gate.active, 1)
        await router.gate.release('busy')

    async def test_api_discovery_errors_and_no_gate_bypass(self):
        async def backend(scope, receive, send): self.fail('Unexpected backend call')
        async def switch(model, protected, active): pass
        router = AgentRouter(backend, route_config(ROLES), switch)
        sent = await self.call(router, path='/v1/models', method='GET')
        self.assertEqual(len(json.loads(sent[-1]['body'])['data']), 7)
        for path, body, status in [('/v1/chat/completions', [], 400),
                                   ('/v1/chat/completions', prompt('hi', 'unknown'), 404),
                                   ('/v1/responses', {}, 404), ('/v1/models/x/load', {}, 404)]:
            sent = await self.call(router, body, path)
            self.assertEqual(sent[0]['status'], status)

    async def test_backend_failure_releases_slot(self):
        async def backend(scope, receive, send): raise RuntimeError('backend failed')
        async def switch(model, protected, active): pass
        router = AgentRouter(backend, route_config(ROLES), switch)
        with self.assertRaises(RuntimeError): await self.call(router, prompt('hi'))
        self.assertEqual(router.gate.active, 0)


class ConcurrentBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_catalogs_allow_coder_and_research_and_3bit_worker(self):
        async def switch(model, protected, active): pass
        routes = route_config(ROLES)
        def gate_for(routes):
            return ModelGate(switch, weights={v['model']: v['weight_bytes'] * 1.05 for v in routes.values()})
        gate = gate_for(routes)
        await gate.acquire(routes['coder']['model'])
        self.assertTrue(gate.fits(routes['research']['model']))
        self.assertFalse(gate.fits(routes['worker']['model']))
        await gate.release(routes['coder']['model'])
        compact = route_config(ROLES, worker_precision='3bit')
        gate = gate_for(compact)
        await gate.acquire(compact['worker']['model'])
        self.assertTrue(gate.fits(compact['coder']['model']))
        self.assertTrue(gate.fits(compact['research']['model']))
        await gate.release(compact['worker']['model'])

    async def test_idle_eviction_never_evicts_active_specialist(self):
        from agent_runtime import reclaim_idle
        from types import SimpleNamespace
        class Pool:
            def __init__(self): self.loaded, self.evicted = {'worker', 'coder'}, []
            def get_loaded_model_ids(self): return list(self.loaded)
            def get_entry(self, model): return SimpleNamespace(last_access=0)
            async def unload_if_idle_unpinned(self, model):
                self.evicted.append(model)
                self.loaded.remove(model)
                return True
        pool = Pool()
        gib = 1024**3
        await reclaim_idle(pool, {'coder', 'research'}, 6*gib,
                           {'worker': 28*gib, 'coder': 16*gib, 'research': 16*gib}, 41*gib)
        self.assertEqual(pool.evicted, ['worker'])
        self.assertIn('coder', pool.loaded)

    async def test_two_distinct_model_streams_remain_active_together(self):
        both_started, finish = asyncio.Event(), asyncio.Event()
        active_models = set()
        async def backend(scope, receive, send):
            active_models.add(json.loads((await receive())['body'])['model'])
            if len(active_models) == 2: both_started.set()
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await finish.wait()
            await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n'})
        async def switch(model, protected, active): pass
        router = AgentRouter(backend, route_config(ROLES), switch)
        caller = MiddlewareTests()
        coder = asyncio.create_task(caller.call(router, prompt('Debug repository')))
        research = asyncio.create_task(caller.call(router, prompt('Research evidence')))
        try:
            await asyncio.wait_for(both_started.wait(), 1)
            self.assertEqual(router.gate.active, 2)
            self.assertEqual(len(router.gate.models), 2)
        finally:
            finish.set()
            await asyncio.gather(coder, research)
        self.assertEqual(router.gate.active, 0)


class ExtractionBudgetTests(unittest.TestCase):
    def test_default_shorter_output_and_explicit_budget_preserved(self):
        routes=route_config(ROLES)
        body=resolve('/v1/chat/completions',prompt('Extract emails'),routes)[1]
        self.assertEqual(body['max_tokens'],2048)
        for name in ('max_tokens','max_completion_tokens'):
            request=prompt('Extract emails');request[name]=6000
            result=resolve('/v1/chat/completions',request,routes)[1]
            self.assertEqual(result[name],6000)
            if name=='max_completion_tokens': self.assertNotIn('max_tokens',result)
