"""Offline authentication policy and pre-deserialization regression checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

FULL_MODEL = Path(__file__).resolve().parents[1] / "full_model"
sys.path.insert(0, str(FULL_MODEL))
import checkpoint_identity as identity


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, FULL_MODEL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name)
        self.pins = {}
        for name in identity.FILES:
            content = name.encode()
            (self.checkpoint / name).write_bytes(content)
            self.pins[name] = hashlib.sha256(content).hexdigest()

    def test_unreviewed_metadata_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "independently trusted.*missing"):
            identity.authenticate(self.checkpoint)

    def test_observations_do_not_claim_authenticated_revision(self):
        with patch.dict(identity.TRUSTED_SHA256,
                        {"model.safetensors": self.pins["model.safetensors"]}, clear=True):
            result = identity.identify(self.checkpoint)
        self.assertFalse(result["checkpoint_authenticated"])
        self.assertNotIn("revision", result)
        self.assertEqual(result["requested_revision"], identity.REVISION)
        self.assertFalse(result["files"]["tokenizer.json"]["authenticated"])
        self.assertTrue(result["files"]["model.safetensors"]["authenticated"])

    def test_forged_local_manifest_cannot_supply_trust(self):
        (self.checkpoint / "checkpoint-identity.json").write_text(json.dumps({
            "checkpoint_authenticated": True, "revision": identity.REVISION,
            "files": self.pins,
        }))
        with self.assertRaisesRegex(RuntimeError, "not trust anchors"):
            identity.authenticate(self.checkpoint)

    def test_complete_independent_test_pins_accept_fixture(self):
        with patch.dict(identity.TRUSTED_SHA256, self.pins, clear=True):
            self.assertTrue(identity.authenticate(self.checkpoint)["checkpoint_authenticated"])

    def test_every_consumed_file_rejects_modified_bytes(self):
        with patch.dict(identity.TRUSTED_SHA256, self.pins, clear=True):
            for name in identity.FILES:
                with self.subTest(name=name):
                    path = self.checkpoint / name
                    content = path.read_bytes()
                    path.write_bytes(content + b"modified")
                    with self.assertRaisesRegex(RuntimeError, name):
                        identity.authenticate(self.checkpoint)
                    path.write_bytes(content)

    def test_loader_refuses_before_transformers_deserialization(self):
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object))
        model = Mock()
        tokenizer = Mock()
        transformers = types.SimpleNamespace(AutoModelForCausalLM=model, AutoTokenizer=tokenizer)
        with patch.dict(sys.modules, torch=torch, transformers=transformers):
            loader = load("qwen_pretrained_test_loader", "loader.py")
            with patch.dict("os.environ", QWEN_CHECKPOINT=str(self.checkpoint)):
                with self.assertRaisesRegex(RuntimeError, "pins are missing"):
                    loader.get_model_and_inputs()
        model.from_pretrained.assert_not_called()
        tokenizer.from_pretrained.assert_not_called()

    def test_file_path_loader_resolves_authentication_without_pythonpath(self):
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object))
        transformers = types.SimpleNamespace(AutoModelForCausalLM=Mock(), AutoTokenizer=Mock())
        search_path = [path for path in sys.path if path != str(FULL_MODEL)]
        with patch.object(sys, "path", search_path), \
             patch.dict(sys.modules, torch=torch, transformers=transformers):
            previous = sys.modules.pop("checkpoint_identity", None)
            try:
                loader = load("qwen_file_path_test_loader", "loader.py")
                self.assertTrue(callable(loader.authenticate))
            finally:
                if previous is not None:
                    sys.modules["checkpoint_identity"] = previous

    def test_driver_refuses_before_export_or_compiler_bootstrap(self):
        driver = load("qwen_pretrained_test_driver", "run.py")
        with patch.object(driver, "bootstrap_compiler") as bootstrap, \
             patch.object(driver, "run") as execute, \
             patch.object(sys, "argv", ["run.py", "--checkpoint", str(self.checkpoint)]):
            with self.assertRaisesRegex(RuntimeError, "pins are missing"):
                driver.main()
        bootstrap.assert_not_called()
        execute.assert_not_called()

    def test_inventory_authenticates_instead_of_trusting_observations(self):
        inventory = load("qwen_test_inventory", "inventory.py")
        with self.assertRaisesRegex(RuntimeError, "pins are missing"):
            inventory.annotate(self.checkpoint / "missing-library", self.checkpoint / "missing-export",
                               self.checkpoint, 32, 2)

    def test_downloader_records_observations_then_refuses_verification(self):
        downloader = load("qwen_test_downloader", "download_checkpoint.py")
        hub = types.SimpleNamespace(snapshot_download=Mock())
        with patch.dict(sys.modules, huggingface_hub=hub), \
             patch.dict(identity.TRUSTED_SHA256,
                        {"model.safetensors": self.pins["model.safetensors"]}, clear=True), \
             patch.object(sys, "argv", ["download_checkpoint.py", "--out", str(self.checkpoint)]):
            with self.assertRaisesRegex(RuntimeError, "pins are missing"):
                downloader.main()
        observed = json.loads((self.checkpoint / "checkpoint-identity.json").read_text())
        self.assertFalse(observed["checkpoint_authenticated"])
        self.assertNotIn("revision", observed)


if __name__ == "__main__":
    unittest.main()
