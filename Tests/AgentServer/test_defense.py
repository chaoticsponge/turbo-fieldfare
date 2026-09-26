import asyncio
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_http import RouteError
from agent_images import validate_images
from agent_routing import resolve
from test_launcher import launcher


def image_messages(count=1, size=(2, 2)):
    from PIL import Image
    encoded = BytesIO()
    Image.new('RGB', size).save(encoded, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(encoded.getvalue()).decode()
    return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}} for _ in range(count)]}]


class DefensiveInputsTests(unittest.TestCase):
    def test_single_model_uses_same_runtime_and_routes_all_auto_prompts_to_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            settings = launcher.settings(root, 2, 8192, 48)
            command = launcher.configure_single(root, root, {'files': []}, settings,
                launcher.command(root, root, 8080, 2, 48), 2, 8192)
            self.assertEqual(Path(command[1]).name, 'agent_runtime.py')
            self.assertNotIn('--model-dir', command)
            config = json.loads((root / 'routes.json').read_text())
            self.assertEqual(config['default_role'], 'worker')
            for text in ('debug this repository', 'research evidence', 'extract names', 'hello'):
                role, _ = resolve('/v1/chat/completions', {'messages': [{'role': 'user', 'content': text}]},
                                  config['routes'], config['default_role'])
                self.assertEqual(role, 'worker')

    def test_image_count_rejected_before_header_or_pixel_decode(self):
        with patch('PIL.Image.open', side_effect=AssertionError('Must not open images')):
            messages = [{'role':'user','content':[{'type':'image_url','image_url':{'url':'unused'}}]*9}]
            with self.assertRaisesRegex(RouteError, 'At most 8'):
                validate_images(messages)

    def test_aggregate_pixels_checked_without_pixel_decode(self):
        from PIL import Image
        messages = image_messages(2)
        with patch.object(Image.Image, 'load', side_effect=AssertionError('Pixel decode before admission')):
            validate_images(copy.deepcopy(messages))
            with patch('agent_images.MAX_PIXELS', 7):
                with self.assertRaisesRegex(RouteError, 'pixel budget'):
                    validate_images(messages)

    def test_image_alias_normalized_and_invalid_sources_fail_closed(self):
        messages = image_messages()
        url = messages[0]['content'][0]['image_url']['url']
        messages[0]['content'] = [{'type':'input_image','input_image':url}]
        validate_images(messages)
        self.assertEqual(messages[0]['content'][0], {'type':'image_url','image_url':{'url':url}})
        messages[0]['content'][0]['image_url']['detail'] = 'low'
        validate_images(messages)
        self.assertEqual(messages[0]['content'][0]['image_url']['detail'], 'low')
        for source in ('https://example.invalid/a.png', '/tmp/a.png', 'file:///tmp/a.png',
                       'data:image/png;base64,!!', 'data:image/jpeg;base64,'+url.split(',')[1]):
            messages[0]['content'][0]['image_url']['url'] = source
            with self.assertRaises(RouteError):validate_images(messages)
        for kind in ('input_audio', 'file', 'video_url', 'unknown'):
            with self.assertRaises(RouteError):validate_images([{'role':'user','content':[{'type':kind}]}])

    def test_encoded_budget_and_non_user_images_rejected(self):
        messages = image_messages()
        with patch('agent_images.MAX_ENCODED_BYTES', 2):
            with self.assertRaises(RouteError):validate_images(messages)
        messages[0]['role'] = 'tool'
        with self.assertRaisesRegex(RouteError, 'user messages'):validate_images(messages)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_pinned_cli_configuration_installs_transport_limits(self):
        import uvicorn
        from agent_transport import AgentHTTPProtocol, install_transport_limits
        original = uvicorn.Config
        try:
            install_transport_limits()
            config = uvicorn.Config('omlx.server:app')
            self.assertIs(config.http, AgentHTTPProtocol)
            self.assertEqual(config.limit_concurrency, 128)
            self.assertEqual(config.ws, 'none')
        finally:
            uvicorn.Config = original

    async def test_header_deadline_and_connection_cap(self):
        import uvicorn
        from uvicorn.server import ServerState
        from agent_transport import AgentHTTPProtocol
        from unittest.mock import MagicMock
        async def app(scope, receive, send):
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'ok'})
        config = uvicorn.Config(app, log_config=None, ws='none')
        config.load()
        state = ServerState()
        def connect():
            transport = MagicMock()
            transport.get_extra_info.return_value = None
            transport.is_closing.return_value = False
            protocol = AgentHTTPProtocol(config, state, {})
            protocol.connection_made(transport)
            return protocol, transport
        with patch('agent_transport.HEADER_TIMEOUT', .01), patch('agent_transport.MAX_CONNECTIONS', 1):
            first, transport = connect()
            second, refused = connect()
            refused.close.assert_called_once()
            first.data_received(b'GET /health HTTP/1.1\r\nHo')
            await asyncio.sleep(.03)
            transport.close.assert_called_once()
            first.connection_lost(None)
            second.connection_lost(None)
            self.assertEqual(len(state.connections), 0)

    async def test_complete_headers_cancel_deadline_and_next_request_rearms_it(self):
        import uvicorn
        from uvicorn.server import ServerState
        from agent_transport import AgentHTTPProtocol
        from unittest.mock import MagicMock
        complete = asyncio.Event()
        async def app(scope, receive, send):
            await complete.wait()
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'ok'})
        config = uvicorn.Config(app, log_config=None, ws='none');config.load()
        transport = MagicMock();transport.get_extra_info.return_value = None
        transport.is_closing.return_value = False
        state = ServerState()
        protocol = AgentHTTPProtocol(config, state, {})
        with patch('agent_transport.HEADER_TIMEOUT', .02):
            protocol.connection_made(transport)
            protocol.data_received(b'GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n')
            await asyncio.sleep(.03)
            transport.close.assert_not_called()  # A long response is not a stalled header.
            complete.set()
            await asyncio.sleep(.01)
            protocol.data_received(b'GET /health HTTP/1.1\r\nHo')
            await asyncio.sleep(.03)
            transport.close.assert_called_once()
            protocol.connection_lost(None)
            await asyncio.gather(*state.tasks)
