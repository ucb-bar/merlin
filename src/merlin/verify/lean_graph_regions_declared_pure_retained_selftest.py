"""Isolated source-data association tests; no Lean/MLIR/native qualification.

Run directly with Python -I -S -B. Temporary data stays in an owned private
profile; no package initializer, SDK/runtime access, fake package or mock.
"""

from __future__ import annotations

import argparse
import json
import runpy
import tempfile
import unittest
from pathlib import Path


class SourceAssociationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.profile)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / self.api["BASE"]
        self.artifacts.mkdir(parents=True)
        self.original = self.artifacts / "source.mlir"
        self.original.write_bytes(b"independent original source bytes\n")
        self.pin = dict(path=str(self.original), **self.api["identity"](self.original.read_bytes()))

    def test_genuine_original(self):
        self.assertEqual(self.api["read_bound"](self.root, self.pin), self.original.read_bytes())

    def test_snapshot_actual_bytes(self):
        target = self.artifacts / "snapshot/source.mlir"
        binding = self.api["copy_data"](self.root, self.pin, target)
        self.assertEqual(self.original.read_bytes(), target.read_bytes())
        self.assertEqual(binding["original"], self.pin)

    def test_changed_original(self):
        self.original.write_bytes(b"swapped source of the same size!\n")
        with self.assertRaises(ValueError):
            self.api["read_bound"](self.root, self.pin)

    def test_missing_original(self):
        self.original.unlink()
        with self.assertRaises(FileNotFoundError):
            self.api["read_bound"](self.root, self.pin)

    def test_boolean_length_refused(self):
        with self.assertRaises(ValueError):
            self.api["read_bound"](self.root, self.pin | dict(bytes=True))

    def test_extra_reference_field_refused(self):
        with self.assertRaises(ValueError):
            self.api["read_bound"](self.root, self.pin | dict(accepted=True))

    def test_outside_source_root_refused(self):
        outside = self.root / "outside.mlir"
        outside.write_bytes(self.original.read_bytes())
        with self.assertRaises(ValueError):
            self.api["read_bound"](self.root, self.pin | dict(path=str(outside)))

    def test_symlink_escape_refused(self):
        outside = self.root / "outside.mlir"
        outside.write_bytes(self.original.read_bytes())
        link = self.artifacts / "escaped.mlir"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.api["read_bound"](self.root, self.pin | dict(path=str(link)))

    def test_coherently_rehashed_replacement_receipt_refused(self):
        receipt = self.artifacts / "receipt.json"
        receipt.write_text(json.dumps(dict(original=self.pin)))
        expected = self.api["identity"](receipt.read_bytes())
        alternate = self.artifacts / "alternate.mlir"
        alternate.write_bytes(b"alternate original\n")
        replacement = dict(path=str(alternate), **self.api["identity"](alternate.read_bytes()))
        receipt.write_text(json.dumps(dict(original=replacement)))
        with self.assertRaises(ValueError):
            self.api["original_evidence"](
                self.root, "receipt.json", (expected["bytes"], expected["sha256"])
            )

    def test_original_fixed_receipt(self):
        receipt = self.artifacts / "receipt.json"
        receipt.write_text(json.dumps(dict(original=self.pin)))
        expected = self.api["identity"](receipt.read_bytes())
        _, document = self.api["original_evidence"](
            self.root, "receipt.json", (expected["bytes"], expected["sha256"])
        )
        self.assertEqual(document["original"], self.pin)

    def test_snapshot_not_overwritten(self):
        target = self.artifacts / "snapshot.mlir"
        target.write_bytes(b"earlier retained snapshot")
        with self.assertRaises(FileExistsError):
            self.api["copy_data"](self.root, self.pin, target)
        self.assertEqual(target.read_bytes(), b"earlier retained snapshot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.profile.mkdir(parents=True, exist_ok=False)
    SourceAssociationTests.api = runpy.run_path(str(arguments.helper.resolve(strict=True)))
    SourceAssociationTests.profile = arguments.profile.resolve(strict=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceAssociationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(
        scope="stdlib-only source-data associations; no compiler/native qualification",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        passed=result.wasSuccessful(),
        native_invocations=0,
        sdk_runtime_reads=0,
    )
    (arguments.profile / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
