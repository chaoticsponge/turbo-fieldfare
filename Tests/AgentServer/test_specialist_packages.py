import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from agent_models import ROOT, ROLES, catalog, model_path
from test_launcher import launcher

spec = importlib.util.spec_from_file_location('agent_installer', ROOT / 'Scripts/manage-agent-models.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class PackageTests(unittest.TestCase):
    def test_catalogs_pin_every_payload_and_no_remote_code(self):
        for role in ROLES:
            data = catalog(role)
            self.assertRegex(data['revision'], r'^[0-9a-f]{40}$')
            names = [f['name'] for f in data['files']]
            self.assertEqual(len(names), len(set(names)))
            self.assertIn('config.json', names)
            self.assertTrue(any(n.endswith('.safetensors') for n in names))
            for f in data['files']:
                self.assertEqual(Path(f['name']).name, f['name'])
                self.assertFalse(f['name'].endswith('.py'))
                self.assertRegex(f['sha256'], r'^[0-9a-f]{64}$')
                self.assertGreater(f['bytes'], 0)

    def test_resume_and_ignored_range_both_verify_before_publish(self):
        data = b'0123456789'
        entry = {'name': 'model.safetensors', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for status in (200, 206):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                (directory / 'model.safetensors.part').write_bytes(data[:3])
                def opener(request, timeout):
                    self.assertEqual(request.get_header('Range'), 'bytes=3-')
                    return Response(data[3:] if status == 206 else data, status,
                                    {'Content-Range': 'bytes 3-9/10'})
                installer.fetch_file(directory, entry, 'https://example.test/file', opener)
                self.assertEqual((directory / entry['name']).read_bytes(), data)
                self.assertFalse((directory / 'model.safetensors.part').exists())

    def test_corrupt_download_never_publishes_final_file(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = {'name': 'model.safetensors', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                installer.fetch_file(Path(folder), entry, 'https://example.test/file',
                                     lambda request, timeout: Response(b'bad'))
            self.assertFalse((Path(folder) / entry['name']).exists())
            self.assertTrue((Path(folder) / (entry['name'] + '.part')).exists())

    def test_fleet_configuration_parses_in_actual_engine(self):
        try:
            from omlx.settings import GlobalSettings
            from omlx.model_settings import ModelSettingsManager
        except ImportError:
            self.skipTest('Install agent dependencies to validate real upstream settings')
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder).resolve()
            for role in ROLES: model_path(role, state).mkdir()
            config = launcher.settings(model_path('worker', state), 2, 65536, 48)
            command = launcher.configure_fleet(state, list(ROLES), state, config,
                launcher.command(model_path('worker', state), state, 8080, 2, 48), 2, 65536)
            (state / 'settings.json').write_text(json.dumps(config))
            parsed = GlobalSettings.load(state)
            self.assertEqual(len(parsed.model.model_dirs), 6)
            self.assertEqual(parsed.scheduler.embedding_batch_size, 4)
            self.assertEqual(parsed.idle_timeout.idle_timeout_seconds, 300)
            self.assertNotIn('--model-dir', command)
            self.assertIn('--host', command)
            self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
            self.assertEqual(command[0], str(launcher.ENGINE.parent / 'python'))
            manager = ModelSettingsManager(state)
            for role, info in ROLES.items():
                settings = manager.get_settings(info['directory'])
                self.assertEqual(settings.model_type_override, info['kind'])
                self.assertFalse(settings.is_pinned)
