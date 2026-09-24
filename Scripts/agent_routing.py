"""Model-free prompt routing with bounded, memory-aware concurrent admission."""
import asyncio
from contextlib import suppress
from contextvars import ContextVar
from collections import deque
import json
import re
from agent_models import weight_reservation
from agent_admission import estimate_session

CHAT_ROLES = {'worker', 'coder', 'research', 'extract'}
PATH_ROLES = {'/v1/embeddings': 'embed', '/v1/rerank': 'rerank'}
# Request-local output reservation consumed by the engine token-count validator.
OUTPUT_RESERVATION = ContextVar('agent_output_reservation', default=0)
TOKEN_RESERVATION = ContextVar('agent_token_reservation', default=None)
MAX_BODY = 16 * 1024 * 1024


class RouteError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def user_intent(messages):
    # Tool results and assistant text never choose the next specialist. This
    # keeps a tool loop on its originating user intent, without stored sessions.
    for message in reversed(messages):
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content', '')
            if isinstance(content, str):
                if re.fullmatch(r'\s*(continue|go on|yes|do it|proceed)[.!]?\s*', content, re.I):
                    continue
                return content
            if isinstance(content, list):
                return '\n'.join(p.get('text', '') for p in content
                                 if isinstance(p, dict) and p.get('type') == 'text'
                                 and isinstance(p.get('text'), str))
    return ''


def has_images(messages):
    return any(isinstance(m, dict) and isinstance(m.get('content'), list)
               and any(isinstance(p, dict) and p.get('type') in ('image_url', 'input_image')
                       for p in m['content']) for m in messages)


def choose_role(body):
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise RouteError('messages must be a nonempty array')
    if has_images(messages):
        return 'worker'
    intent = user_intent(messages)
    text = intent[:4000].lower()
    if re.search(r'\b(debug|refactor|implement|codebase|repository|pull request|unit tests?|stack trace|traceback|git diff)\b|\b(write|fix|build|review)\b.{0,60}\b(code|script|function|bug|test|scraper|api|server|docker|terraform)\b', text):
        return 'coder'
    if re.search(r'\b(research|literature review|investigate|compare sources|fact.check|evidence|citations?|whitepaper)\b', text):
        return 'research'
    if re.search(r'\b(extract|classify|classification|categorize|scrape|scraping|parse|deduplicate|label)\b', text):
        return 'extract'
    if len(intent) > 24000:
        return 'research'
    return 'worker'


def resolve(path, body, routes):
    if not isinstance(body, dict):
        raise RouteError('Expected a JSON object')
    model = body.get('model', 'auto')
    if not isinstance(model, str):
        raise RouteError('model must be a string')
    expected = PATH_ROLES.get(path)
    if model == 'auto':
        role = expected or choose_role(body)
    elif model in routes:
        role = model
    else:
        role = next((r for r, config in routes.items() if config['model'] == model), None)
        if role is None:
            raise RouteError(f'Unknown or unavailable model: {model}', 404)
    if expected and role != expected or not expected and role not in CHAT_ROLES:
        raise RouteError('Model role is not compatible with this endpoint')
    if role not in routes:
        raise RouteError(f"Role '{role}' is not installed/enabled; install it or choose an enabled role explicitly", 503)
    config = routes[role]
    if path == '/v1/chat/completions':
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise RouteError('messages must be a nonempty array')
        if has_images(messages) and config['kind'] != 'vlm':
            raise RouteError('This specialist cannot accept images; use worker or auto')
        limit = config.get('max_context_window')
        if limit is not None:
            supplied = [body[k] for k in ('max_tokens', 'max_completion_tokens') if body.get(k) is not None]
            if any(type(v) is not int or v <= 0 for v in supplied):
                raise RouteError('Output token budget must be a positive integer')
            if len(set(supplied)) > 1:
                raise RouteError('max_tokens and max_completion_tokens must agree')
            output = supplied[0] if supplied else min(2048 if role == 'extract' else 4096, limit // 2)
            if output >= limit:
                raise RouteError(f"Role '{role}' has a {limit}-token context budget; output must leave room for the prompt")
            body['max_tokens'] = output
        for key in ('temperature', 'top_p', 'top_k'):
            body.setdefault(key, config[key])
        if role == 'extract':
            if 'max_tokens' not in body and 'max_completion_tokens' not in body:
                body['max_tokens'] = 2048
            kwargs = body.setdefault('chat_template_kwargs', {})
            if not isinstance(kwargs, dict):
                raise RouteError('chat_template_kwargs must be an object')
            kwargs.setdefault('enable_thinking', False)
    elif expected == 'embed':
        inputs = body.get('input')
        inputs = [inputs] if isinstance(inputs, str) else inputs
        if (not isinstance(inputs, list) or not 1 <= len(inputs) <= 32
                or any(not isinstance(v, str) or not v or len(v) > 8192 for v in inputs)):
            raise RouteError('Embedding input must be 1–32 nonempty strings of at most 8192 characters each')
    elif expected == 'rerank':
        docs, query = body.get('documents'), body.get('query')
        if (not isinstance(query, str) or not query or len(query) > 8192
                or not isinstance(docs, list) or not 1 <= len(docs) <= 32
                or any(not isinstance(v, str) or not v or len(v) > 16384 for v in docs)):
            raise RouteError('Rerank requires a query up to 8192 characters and 1–32 text documents up to 16384 characters each')
    body['model'] = config['model']
    return role, body


class ModelGate:
    """Reserve shared weights once and session headroom per active request.

    Different models may run concurrently when their reservations fit. Idle
    engines can be reclaimed by the switch callback; active models are protected.
    Reservations last until the entire ASGI response (including SSE) finishes.
    Estimates supplement, rather than replace, oMLX's actual-memory guard.
    """
    def __init__(self, switch, concurrency=2, max_waiters=16, weights=None,
                 budget_bytes=40.8 * 1024**3, session_bytes=3 * 1024**3):
        self.switch = switch
        self.concurrency = concurrency
        self.max_waiters = max_waiters
        self.condition = asyncio.Condition()
        self.waiters = deque()
        self.active = 0
        self.models = {}
        self.weights = weights or {}
        self.budget = budget_bytes
        self.session_bytes = session_bytes
        self.active_session_bytes = 0
        self.leases = {}

    def fits(self, model, session_bytes=None):
        size = self.session_bytes if session_bytes is None else session_bytes
        weights = sum(self.weights.get(m, 0) for m in {*self.models, model})
        return weights + self.active_session_bytes + size <= self.budget

    async def acquire(self, model, session_bytes=None):
        size = self.session_bytes if session_bytes is None else session_bytes
        if type(size) is not int or size <= 0:
            raise RouteError('Invalid session memory reservation')
        ticket = object()
        async with self.condition:
            if self.weights.get(model, 0) + size > self.budget:
                raise RouteError('Model and requested context cannot fit the admission budget; reduce history or output tokens', 503)
            if len(self.waiters) >= self.max_waiters:
                raise RouteError('Agent queue is full; retry later', 429)
            self.waiters.append(ticket)
            try:
                await self.condition.wait_for(lambda:
                    self.waiters[0] is ticket and self.active < self.concurrency
                    and self.fits(model, size))
                await self.switch(model, {*self.models, model}, self.active_session_bytes + size)
                self.active += 1
                self.active_session_bytes += size
                self.leases[ticket] = (model, size)
                self.models[model] = self.models.get(model, 0) + 1
                return ticket
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    async def release(self, lease):
        async with self.condition:
            # Legacy callers releasing a model used equal fixed reservations.
            # Request paths always release the exact lease, including out of order.
            if isinstance(lease, str):
                lease = next(key for key, (model, _) in self.leases.items() if model == lease)
            model, size = self.leases.pop(lease)
            self.active_session_bytes -= size
            self.active -= 1
            self.models[model] -= 1
            if not self.models[model]:
                del self.models[model]
            self.condition.notify_all()


async def reply(send, status, body, headers=()):
    raw = json.dumps(body).encode()
    await send({'type': 'http.response.start', 'status': status,
                'headers': [(b'content-type', b'application/json'),
                            (b'content-length', str(len(raw)).encode()), *headers]})
    await send({'type': 'http.response.body', 'body': raw})


class AgentRouter:
    def __init__(self, app, routes, switch, concurrency=2, budget_bytes=40.8 * 1024**3, cache_status=None, prefix_status=None, read_ahead_status=None):
        self.app, self.routes = app, routes
        self.cache_status = cache_status
        self.prefix_status = prefix_status
        self.read_ahead_status = read_ahead_status
        self.gate = ModelGate(switch, concurrency, budget_bytes=budget_bytes,
                             weights={entry['model']: weight_reservation(entry)
                                      for entry in routes.values()})

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            return await self.app(scope, receive, send)
        if scope['type'] != 'http':
            return await send({'type': 'websocket.close', 'code': 1008})
        path, method = scope['path'], scope['method']
        if method == 'GET' and path == '/health':
            return await reply(send, 200, {'status': 'ok', 'routing': 'prompt-rules', 'roles': list(self.routes),
                'expert_read_ahead': self.read_ahead_status() if self.read_ahead_status else None,
                'prefix_cache': self.prefix_status() if self.prefix_status else {'enabled': False},
                'adaptive_expert_cache': self.cache_status() if self.cache_status else None,
                'admission': {'mode': 'token-aware', 'active_requests': self.gate.active,
                    'active_session_bytes': self.gate.active_session_bytes,
                    'models': {role: entry.get('cache_profile', {'layout': 'fallback'})
                               for role, entry in self.routes.items()}},
                'context_budgets': {role: entry['max_context_window'] for role, entry in self.routes.items()
                                    if 'max_context_window' in entry},
                'expert_streaming': {role: {'cache_bytes': entry['expert_streaming']['cache_bytes'],
                    'planned_weight_bytes': entry['resident_weight_bytes']}
                    for role, entry in self.routes.items() if 'expert_streaming' in entry}})
        if method == 'GET' and path == '/v1/models':
            data = [{'id': r, 'object': 'model', 'owned_by': 'local',
                     'created': 0, 'underlying_model': c['model'],
                     **({'max_context_window': c['max_context_window']} if 'max_context_window' in c else {})} for r, c in self.routes.items()]
            data.insert(0, {'id': 'auto', 'object': 'model', 'created': 0, 'owned_by': 'local'})
            return await reply(send, 200, {'object': 'list', 'data': data})
        # No alternate inference/admin routes can bypass the shared memory gate.
        if method != 'POST' or path not in {'/v1/chat/completions', *PATH_ROLES}:
            return await reply(send, 404, {'error': {'message': 'Unsupported routed endpoint'}})
        try:
            raw = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                raw.extend(message.get('body', b''))
                if len(raw) > MAX_BODY:
                    raise RouteError('Request exceeds 16 MiB', 413)
                if not message.get('more_body', False):
                    break
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                raise RouteError('Invalid JSON')
            role, body = resolve(path, body, self.routes)
            estimate = estimate_session(path, body, self.routes[role])
            encoded = json.dumps(body).encode()
            scope = {**scope, 'headers': [(k, v) for k, v in scope.get('headers', [])
                                         if k.lower() not in (b'content-length', b'transfer-encoding')]
                     + [(b'content-length', str(len(encoded)).encode())]}
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': encoded, 'more_body': False}
                return await receive()

            async def report(message):
                if message['type'] == 'http.response.start':
                    message = {**message, 'headers': [*message.get('headers', []),
                        (b'x-agent-role', role.encode()), (b'x-agent-model', body['model'].encode()),
                        (b'x-agent-session-bytes', str(estimate.session_bytes).encode()),
                        (b'x-agent-token-estimate', str(estimate.total_tokens or 0).encode()),
                        (b'x-agent-estimate-method', estimate.method.encode())]}
                await send(message)

            async def disconnect():
                while (await receive())['type'] != 'http.disconnect':
                    pass

            admission = asyncio.create_task(self.gate.acquire(body['model'], estimate.session_bytes))
            gone = asyncio.create_task(disconnect())
            try:
                done, _ = await asyncio.wait([admission, gone], timeout=300,
                                             return_when=asyncio.FIRST_COMPLETED)
                if gone in done:
                    return
                if admission not in done:
                    raise RouteError('Timed out waiting for a model slot; retry later', 503)
                await admission
                gone.cancel()
                with suppress(asyncio.CancelledError):
                    await gone
                reservation = OUTPUT_RESERVATION.set(body.get('max_tokens', 0)
                    if path == '/v1/chat/completions' and 'max_context_window' in self.routes[role] else 0)
                token_reservation = TOKEN_RESERVATION.set(estimate.total_tokens)
                try:
                    await self.app(scope, replay, report)
                finally:
                    TOKEN_RESERVATION.reset(token_reservation)
                    OUTPUT_RESERVATION.reset(reservation)
            finally:
                gone.cancel()
                with suppress(asyncio.CancelledError):
                    await gone
                if not admission.done():
                    admission.cancel()
                with suppress(asyncio.CancelledError, RouteError):
                    await admission
                if not admission.cancelled() and admission.exception() is None:
                    await self.gate.release(admission.result())
        except RouteError as error:
            await reply(send, error.status, {'error': {'message': str(error), 'type': 'routing_error'}})
