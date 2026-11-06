"""Pinned library admission uses Git blobs even when HEAD is unchanged."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).absolute().parents[3] / "examples/atlas/model_bringup/source_view.py"
SPEC = importlib.util.spec_from_file_location("atlas_model_source_view", MODULE)
source_view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source_view)


class SourceViewPinTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.atlas = self.root / "atlas"
        self.merlin = self.root / "merlin"
        self.kernels = self.root / "kernels"
        for path in (self.atlas, self.merlin, self.kernels):
            path.mkdir()
        self.kernel_name = "lib/kernel.py"
        self.kernel = self.write(self.kernels, self.kernel_name, b"PINNED = True\n")
        self.write(self.atlas, "runtime/common.h", b"/* compiler */\n")
        self.write(self.atlas, "runtime/model.c", b"/* baseline model */\n")
        self.write(self.merlin, "examples/atlas/model_bringup/runtime/model.c", b"/* model */\n")
        self.git("init", "--quiet")
        self.git("add", "--", self.kernel_name)
        self.git("-c", "user.name=Nicolas Reed", "-c", "user.email=reednicolas@berkeley.edu",
                 "commit", "--quiet", "-m", "test: pin fixture")
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.map = self.root / "source-map.json"
        self.map.write_text(json.dumps(dict(
            schema_version=2, library_revision=self.revision,
            sources={"atlas-mlir": ["runtime/common.h"], "merlin": ["runtime/model.c"],
                     "atlas-kernels": [self.kernel_name]}, new_paths=[])))

    @staticmethod
    def write(root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.kernels), *args], text=True)

    def build(self, view, mode="relocated"):
        return source_view.build_view(self.atlas, self.merlin, self.kernels, view,
                                      mode=mode, source_map=self.map)

    def verify(self, view, mode="relocated"):
        return source_view.verify_view(view, self.atlas, self.merlin, self.kernels, mode)

    def test_clean_pinned_library_builds_and_verifies_in_both_modes(self):
        for mode in ("baseline", "relocated"):
            with self.subTest(mode=mode):
                view = self.build(self.root / mode, mode)
                receipt = self.verify(view, mode)
                self.assertEqual(receipt["library_revision"], self.revision)
                self.assertEqual((view / source_view.LIBRARY / self.kernel_name).read_bytes(),
                                 self.kernel.read_bytes())
                self.assertEqual(len(receipt["entries"]), 3)

    def test_dirty_library_bytes_are_refused_before_view_creation(self):
        self.kernel.write_bytes(b"PINNED = False\n")
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.revision)
        for mode in ("baseline", "relocated"):
            with self.subTest(mode=mode):
                view = self.root / mode
                with self.assertRaisesRegex(ValueError, "library source differs from admitted pin"):
                    self.build(view, mode)
                self.assertFalse(view.exists())
                self.assertFalse(source_view.inventory_path(view).exists())

    def test_staged_index_cannot_replace_the_pinned_blob(self):
        self.kernel.write_bytes(b"PINNED = False\n")
        self.git("add", "--", self.kernel_name)
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.revision)
        with self.assertRaisesRegex(ValueError, "library source differs from admitted pin"):
            self.build(self.root / "view")

    def test_refreshed_receipt_cannot_admit_dirty_library_bytes(self):
        view = self.build(self.root / "view")
        self.kernel.write_bytes(b"PINNED = False\n")
        (view / source_view.LIBRARY / self.kernel_name).write_bytes(self.kernel.read_bytes())
        receipt_path = source_view.inventory_path(view)
        receipt = json.loads(receipt_path.read_text())
        for entry in receipt["entries"]:
            if entry["owner"] == "atlas-kernels":
                entry["sha256"] = source_view.digest(self.kernel)
        receipt["inventory_sha256"] = source_view.inventory_digest(receipt["entries"])
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.revision)
        with self.assertRaisesRegex(ValueError, "library source differs from admitted pin"):
            self.verify(view)

    def test_inventory_revision_must_match_the_source_map_pin(self):
        view = self.build(self.root / "view")
        receipt_path = source_view.inventory_path(view)
        receipt = json.loads(receipt_path.read_text())
        receipt["library_revision"] = "0" * 40
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "inventory library revision differs from admitted pin"):
            self.verify(view)

    def test_unlisted_dirty_files_do_not_expand_the_execution_view(self):
        self.write(self.kernels, "unlisted.py", b"UNLISTED = True\n")
        view = self.build(self.root / "view")
        self.verify(view)
        self.assertFalse((view / source_view.LIBRARY / "unlisted.py").exists())


if __name__ == "__main__":
    unittest.main()
