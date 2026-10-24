"""Pure controls for native-example input admission, without workload execution."""

from merlin.common.digest import sha256_bytes

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from merlin.common.paths import repo_root


path = repo_root() / "examples/atlas/comparisons/capture_tinyllama_native.py"
spec = importlib.util.spec_from_file_location("native_admission_controls", path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def member(name, mode=0):
    return SimpleNamespace(filename=name, external_attr=mode << 16)


class NativeAdmissionTest(unittest.TestCase):
    def test_resnet_delegates_strict_admission_and_retains_array_hashing(self):
        path = repo_root() / "examples/atlas/comparisons/refresh/capture_resnet.py"
        spec = importlib.util.spec_from_file_location("native_resnet_admission_controls", path)
        resnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet)
        with patch.object(capture.os, "lstat", side_effect=AssertionError("filesystem access")):
            with self.assertRaises(ValueError):
                resnet.safe_path("/not-opened/../destination")
        values = capture.np.arange(4, dtype=capture.np.float32)

        class Tensor:
            def detach(self):
                return self

            cpu = contiguous = detach

            def numpy(self):
                return values

        self.assertEqual(resnet.tensor_identity(Tensor())["raw_sha256"],
                         sha256_bytes(values.tobytes()))

    def test_archive_metadata_refuses_paths_duplicates_links(self):
        capture.admit_archive_members([member("checkpoint/"), member("checkpoint/data/0")])
        bad = ([member("../x")], [member("/x")], [member("a\\b")],
               [member("a"), member("a")], [member("link", stat.S_IFLNK)],
               [SimpleNamespace(filename="safe", orig_filename="safe\0hidden", external_attr=0)])
        for members in bad:
            with self.subTest(members=members), self.assertRaises(ValueError):
                capture.admit_archive_members(members)

    def test_archive_refusal_precedes_content_hash(self):
        archive = SimpleNamespace(infolist=lambda: [member("../x")])
        with patch.object(capture, "safe_path", return_value=Path("/not-opened/archive.npz")), \
                patch.object(capture.zipfile, "is_zipfile", return_value=True), \
                patch.object(capture.zipfile, "ZipFile") as opened, \
                patch.object(Path, "open", side_effect=AssertionError("content was read")):
            opened.return_value.__enter__.return_value = archive
            with self.assertRaises(ValueError):
                capture.digest("/not-opened/archive.npz")

    def test_output_root_is_strict_and_accepts_only_descendants(self):
        with patch.object(capture, "safe_path", side_effect=Path), \
                patch.object(capture, "artifacts_dir", return_value=Path("/not-opened/artifacts")):
            self.assertEqual(capture.artifact_output("/not-opened/artifacts/run"),
                             Path("/not-opened/artifacts/run"))
            for value in ("/not-opened/artifacts", "/not-opened/artifacts-other/run"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    capture.artifact_output(value)

    def provider(self, root):
        tools = root / "tools"
        tools.mkdir()
        sources = {
            "atlas_tile_arithmetic.py": "VALUE = 7\n",
            "run_model_tile.py": "from atlas_tile_arithmetic import VALUE\n",
            "run_model_contraction.py": "# source-bound but never imported here\n",
            "validate_model_contraction.py": (
                "from run_model_tile import VALUE\n"
                "def bind(*args, **kwargs): pass\n"
                "def admit_contraction(*args, **kwargs): pass\n"
                "def recheck(records):\n"
                "    for record in records:\n"
                "        if sha256(record['path']) != record['sha256']:\n"
                "            raise ValueError('binding changed')\n"),
        }
        for name, source in sources.items():
            (tools / name).write_text(source)
        code = {name: sha256_bytes(source.encode()) for name, source in sources.items()
                if name != "validate_model_contraction.py"}
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps(dict(status="passed_model_only", code_before=code, code_after=code)))
        return tools, receipt, capture.digest(receipt), capture.digest(tools / "validate_model_contraction.py")

    def test_all_sources_are_authenticated_before_any_provider_code_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools, receipt, receipt_sha, validator_sha = self.provider(root)
            (tools / "run_model_tile.py").write_text("raise AssertionError('must never execute')\n")
            with patch.object(capture.importlib.util, "spec_from_file_location") as imported:
                with self.assertRaisesRegex(ValueError, "Source hash differs"):
                    capture.load_provider_validator(root, validator_sha, receipt, receipt_sha)
                imported.assert_not_called()

    def test_private_validator_rechecks_loaded_sources_and_uses_strict_io(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            for name in ("run_model_tile", "atlas_tile_arithmetic"):
                sys.modules.pop(name, None)
            root = Path(directory)
            tools, receipt, receipt_sha, validator_sha = self.provider(root)
            validator, record = capture.load_provider_validator(root, validator_sha, receipt, receipt_sha)
            self.assertEqual(validator.VALUE, 7)
            self.assertEqual(record["sha256"], validator_sha)
            self.assertIs(validator.sha256, capture.digest)
            self.assertIs(validator.physical_path, capture.safe_path)
            validator.recheck([])
            # Reuse only an already bound helper from the exact same epoch/root.
            capture.load_provider_validator(root, validator_sha, receipt, receipt_sha)
            (tools / "atlas_tile_arithmetic.py").write_text("VALUE = 8\n")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                validator.recheck([])

    def test_unbound_cached_helpers_and_missing_explicit_pins_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            root = Path(directory)
            _, receipt, receipt_sha, validator_sha = self.provider(root)
            sys.modules["atlas_tile_arithmetic"] = SimpleNamespace()
            with self.assertRaisesRegex(ValueError, "Cached provider helper"):
                capture.load_provider_validator(root, validator_sha, receipt, receipt_sha)
            with self.assertRaisesRegex(ValueError, "explicit validator and receipt"):
                capture.load_provider_validator(root, None, receipt, receipt_sha)

    def test_current_workload_tuple_requires_archived_producer_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, receipt, _, validator_sha = self.provider(root)
            metadata = json.loads(receipt.read_text())
            pins = metadata["code_before"]
            pins["run_workload_contraction.py"] = pins.pop("run_model_contraction.py")
            receipt.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(
                ValueError,
                r"archived Atlas cold-contraction producer root.*run_model_contraction\.py.*"
                r"validate_model_contraction\.py.*bind/recheck/admit_contraction.*no longer ships",
            ):
                capture.load_provider_validator(root, validator_sha, receipt, capture.digest(receipt))


if __name__ == "__main__":
    unittest.main()
