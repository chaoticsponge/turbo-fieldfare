"""Exercise the real upstream ASGI application without lifespan/model startup."""
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from omlx import server
from agent_routing import AgentRouter
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('boundary_launcher', ROOT/'Scripts/serve-qwen-agents.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class APIBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_launch_modes_block_upstream_admin_and_foreign_hosts(self):
        async def switch(*args):
            self.fail('No model admission expected')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text('{}')
            configured = launcher.settings(root, 2, 8192, 48)
            command = launcher.command(root, root, 8080, 2, 48)
            single = launcher.configure_single(root, root, {'files':[]}, configured, command, 2, 8192)
            single_config = json.loads((root/'routes.json').read_text())
            fleet = launcher.configure_fleet(root, ['worker'], root, configured, command, 2, 8192)
            fleet_config = json.loads((root/'routes.json').read_text())
            for argv, config in ((single, single_config), (fleet, fleet_config)):
                self.assertEqual(Path(argv[1]).name, 'agent_runtime.py')
                app = AgentRouter(server.app, config['routes'], switch, default_role=config['default_role'])
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8080') as client:
                    for path in ('/admin/api/setup-api-key', '/v1/completions', '/v1/models/unload', '/mcp'):
                        response = await client.post(path, json={})
                        self.assertEqual(response.status_code, 404, (path, response.text))
                    response = await client.post('/admin/api/setup-api-key', json={},
                        headers={'Host':'audit.invalid:8080','Origin':'http://audit.invalid:8080'})
                    self.assertEqual(response.status_code, 403)
                    self.assertNotIn('set-cookie', response.headers)
                    response = await client.get('/health')
                    self.assertEqual(response.status_code, 200)
                    response = await client.post('/v1/chat/completions', json={
                        'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'unused'}}]*9}]})
                    self.assertIn(response.status_code, (400, 413))  # llm refuses images; vlm refuses count.
