import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen_agents", ROOT / "Scripts/serve-qwen-agents.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class AgentLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        payload = b"small fixture, not model weights"
        (self.root / "model.safetensors").write_bytes(payload)
        self.catalog = {"repoID": "test/model", "revision": "a" * 40, "files": [
            {"name": "model.safetensors", "bytes": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest()}]}
        (self.root / "qwen-install.json").write_text(json.dumps(self.catalog))

    def tearDown(self):
        self.temporary.cleanup()

    def test_verified_installation_and_tampered_payload(self):
        self.assertEqual(launcher.validate_package(self.root, self.catalog),
                         self.catalog["files"][0]["bytes"])
        path = self.root / "model.safetensors"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "checksum"):
            launcher.validate_package(self.root, self.catalog)

    def test_wrong_receipt_and_missing_file(self):
        (self.root / "qwen-install.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "pinned"):
            launcher.validate_package(self.root, self.catalog)
        (self.root / "qwen-install.json").write_text(json.dumps(self.catalog))
        (self.root / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            launcher.validate_package(self.root, self.catalog)

    def test_additional_weights_and_nested_models_are_refused(self):
        extra = self.root / "extra.safetensors"
        extra.write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            launcher.validate_package(self.root, self.catalog)
        extra.unlink()
        (self.root / "another-model").mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            launcher.validate_package(self.root, self.catalog)

    def test_symlink_payload_is_refused(self):
        path = self.root / "model.safetensors"
        target = self.root / "target"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            launcher.validate_package(self.root, self.catalog)

    def test_small_machine_refused_and_64gb_has_headroom(self):
        weights = 30_000_000_000
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            launcher.memory_ceiling(18 * launcher.GIB, weights)
        self.assertEqual(launcher.memory_ceiling(64 * launcher.GIB, weights), 48)

    def test_existing_process_or_failed_inspection_blocks_launch(self):
        for code, output in [(0, "123 TurboFieldfareMac"), (2, "")]:
            with patch.object(launcher.subprocess, "run", return_value=
                              subprocess.CompletedProcess([], code, output, "error")):
                with self.assertRaises(ValueError):
                    launcher.check_processes()
        with patch.object(launcher.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 1, "", "")):
            launcher.check_processes()

    def test_settings_parse_in_pinned_engine_without_loading_model(self):
        try:
            from omlx.settings import GlobalSettings
        except ImportError:
            self.skipTest("Install the pinned server environment to validate upstream settings")
        (self.root / "settings.json").write_text(json.dumps(
            launcher.settings(self.root, 2, 16384, 48)))
        with patch.dict("os.environ", {}, clear=True):
            settings = GlobalSettings.load(base_path=self.root)
        self.assertEqual(settings.validate(), [])
        self.assertEqual(settings.scheduler.max_concurrent_requests, 2)
        self.assertTrue(settings.scheduler.chunked_prefill)
        self.assertEqual(settings.sampling.max_context_window_policy, 16384)
        self.assertFalse(settings.cache.enabled)
        self.assertFalse(settings.huggingface.hf_cache_enabled)
        self.assertEqual(settings.get_effective_model_dirs(), [self.root.resolve()])

    def test_bounded_prefix_cache_uses_ssd_without_a_hot_ram_cache(self):
        try:
            from omlx.settings import GlobalSettings
        except ImportError:
            self.skipTest("Install the pinned server environment to validate upstream settings")
        directory = self.root / "prefix-cache"
        (self.root / "settings.json").write_text(json.dumps(
            launcher.settings(self.root, 2, 65536, 48, 4, directory)))
        with patch.dict("os.environ", {}, clear=True):
            settings = GlobalSettings.load(base_path=self.root)
        self.assertEqual(settings.validate(), [])
        self.assertTrue(settings.cache.enabled)
        self.assertEqual(settings.cache.ssd_cache_max_size, "4GB")
        self.assertEqual(settings.cache.hot_cache_max_size, "0")
        self.assertEqual(settings.cache.initial_cache_blocks, 4)
        args = launcher.command(self.root, self.root, 8080, 2, 48, 4, directory)
        self.assertNotIn("--no-cache", args)
        self.assertEqual(args[args.index("--paged-ssd-cache-dir") + 1], str(directory))

    def test_every_bundled_qwen_receipt_is_recognized_but_modified_receipts_are_not(self):
        catalogs = launcher.supported_catalogs()
        self.assertEqual(len(catalogs), 6)
        for catalog in catalogs:
            (self.root / "qwen-install.json").write_text(json.dumps(catalog))
            self.assertEqual(launcher.installed_catalog(self.root), catalog)
        catalog = dict(catalogs[0], revision="0" * 40)
        (self.root / "qwen-install.json").write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, "supported pinned"):
            launcher.installed_catalog(self.root)

    def test_launch_argv_does_not_use_shell_or_scan_other_models(self):
        path = self.root / "model with spaces"
        args = launcher.command(path, self.root, 8080, 2, 48)
        self.assertEqual(args[args.index("--model-dir") + 1], str(path))
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertIn("--no-hf-cache", args)
        self.assertIn("--no-cache", args)


if __name__ == "__main__":
    unittest.main()
