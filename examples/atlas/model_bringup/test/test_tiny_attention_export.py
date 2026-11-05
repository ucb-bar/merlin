"""Attention exporter provenance and explicit physical platform admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools/tiny_full"))
import attention_export as exporter


class AttentionExportTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TINY_ATTENTION_CAPTURE") and os.environ.get("TINY_MODEL_ROOT"),
                         "set explicit eight-launch diagnostic capture and model fixtures")
    def test_eight_launch_cli_original_numerics_remain_diagnostic(self):
        from attention_bundle import validate_gate
        capture = exporter.checked(os.environ["TINY_ATTENTION_CAPTURE"])
        model = exporter.checked(os.environ["TINY_MODEL_ROOT"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "qualified"
            subprocess.run([sys.executable, "-B", str(exporter.ROOT / "tools/tiny_full/export_capture.py"),
                "--capture", str(capture), "--capture-sha256", exporter.sha256(capture),
                "--model-root", str(model), "--out", str(out), "--dram-bytes", "0x80000000",
                "--qualify-launches"], check=True, capture_output=True, text=True)
            data = json.loads((out / "contract.json").read_text())
            gate = json.loads((out / "qualification.json").read_text())
            sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                    for line in (out / "SHA256SUMS").read_text().splitlines()}
            self.assertEqual(data["kernel_count"], 8)
            self.assertEqual(gate["status"], "PASS")
            self.assertEqual(gate["model_pin"], exporter.MODEL_PIN)
            self.assertEqual(gate["numerics"], "rtl")
            self.assertTrue(gate["all_launches_original_npu_model_rtl_exact"])
            self.assertEqual(gate["verified_indices"], list(range(8)))
            self.assertEqual(sums["qualification.json"], exporter.sha256(out / "qualification.json"))
            for launch in gate["launches"]:
                self.assertEqual(launch["status"], "PASS")
                self.assertTrue(launch["exact"] and launch["full_arena_exact"])
                self.assertFalse(launch["accelerated"] or launch["native_dispatch_enabled"])
                self.assertGreaterEqual(len(launch["source_sha256"]), 45)
                self.assertEqual(launch["model"]["commit"], exporter.MODEL_PIN)
                self.assertEqual(launch["model"]["root"], str(model))
                for filename, expected in launch["source_sha256"].items():
                    self.assertEqual(exporter.sha256(exporter.checked(filename)), expected)
            # This remains an actual eight-launch original-model execution test,
            # but neither of the tracked builder's two complete workload scopes.
            with self.assertRaisesRegex(ValueError, "complete qualification scope"):
                validate_gate(gate, data, sums["replay.json"], sums)

    def test_archived_codec_is_executed_instead_of_changed_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "tiny_fp8_pack.py"
            original.write_text("raise RuntimeError('changed current codec executed')\n")
            archived = root / "captured_codec.py"
            archived.write_text("def encode(words):\n    return b'captured bytes', -7\n")
            expected = hashlib.sha256(archived.read_bytes()).hexdigest()
            capture = root / "capture.json"
            capture.write_text(json.dumps({"source_records": [{"path": str(original), "sha256": expected}]}))
            archive = root / "archive.json"
            archive.write_text(json.dumps({"schema": "atlas.capture_source_archive.v1", "sources": [
                {"original_path": str(original), "sha256": expected, "file": archived.name}]}))
            captured = exporter.Capture(capture, hashlib.sha256(capture.read_bytes()).hexdigest(), archive)
            codec = exporter.capture_codec(captured)
            self.assertEqual(codec.encode(None), (b"captured bytes", -7))
            self.assertEqual(Path(codec.__file__), archived)
            self.assertEqual(captured.pins[str(archived)], expected)
            self.assertTrue(all(exporter.sha256(Path(path)) == value for path, value in captured.pins.items()))
            archived.write_text("def encode(words):\n    return b'changed archive', 0\n")
            self.assertTrue(any(exporter.sha256(Path(path)) != value for path, value in captured.pins.items()))
            with self.assertRaisesRegex(ValueError, "without an exact archived copy"):
                exporter.capture_codec(captured)

    def test_missing_original_can_use_an_exact_archived_codec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "absent" / "tiny_fp8_pack.py"
            archived = root / "captured_codec.py"
            archived.write_text("def encode(words):\n    return b'archived missing source', 3\n")
            expected = hashlib.sha256(archived.read_bytes()).hexdigest()
            receipt = root / "capture.json"
            receipt.write_text(json.dumps({"source_records": [{"path": str(original), "sha256": expected}]}))
            archive = root / "archive.json"
            archive.write_text(json.dumps({"schema": "atlas.capture_source_archive.v1", "sources": [
                {"original_path": str(original), "sha256": expected, "file": archived.name}]}))
            captured = exporter.Capture(receipt, hashlib.sha256(receipt.read_bytes()).hexdigest(), archive)
            self.assertEqual(exporter.capture_codec(captured).encode(None), (b"archived missing source", 3))
            self.assertIsNone(captured.source_substitutions[0]["current_original_sha256"])

    def test_explicit_source_relocation_preserves_receipt_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moved = root / "new"
            moved.mkdir()
            codec = moved / "tiny_fp8_pack.py"
            codec.write_text("def encode(words):\n    return b'moved codec', 5\n")
            original_root = root / "absent"
            original = original_root / codec.name
            expected = hashlib.sha256(codec.read_bytes()).hexdigest()
            receipt = root / "capture.json"
            receipt.write_text(json.dumps({"source_records": [{"path": str(original), "sha256": expected}]}))
            capture_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
            captured = exporter.Capture(receipt, capture_sha, relocations=[f"{original_root}={moved}"])
            self.assertEqual(exporter.capture_codec(captured).encode(None), (b"moved codec", 5))
            self.assertEqual(captured.pins[str(receipt)], capture_sha)
            self.assertEqual(captured.pins[str(codec)], expected)
            self.assertEqual(captured.data["source_records"][0]["path"], str(original))
            codec.write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "without an exact archived copy"):
                exporter.Capture(receipt, capture_sha, relocations=[f"{original_root}={moved}"])

    def test_relocation_syntax_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "capture.json"
            receipt.write_text(json.dumps({"source_records": []}))
            expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
            for relocation in ("missing-separator", "relative=/allowed"):
                with self.subTest(relocation=relocation):
                    with self.assertRaises(ValueError):
                        exporter.Capture(receipt, expected, relocations=[relocation])

    def test_platform_range_covers_every_launch(self):
        args = argparse.Namespace(dram_base=0x80000000, dram_bytes=0x80000000)
        launches = [{"dram_base": exporter.ARENA_BASE, "total_dram_bytes": 1188480}]
        self.assertEqual(exporter.platform_placement(args, launches),
                         {"arena_base": exporter.ARENA_BASE, "dram_base": 0x80000000, "dram_bytes": 0x80000000})
        args.dram_bytes = exporter.ARENA_BASE - args.dram_base + 64
        with self.assertRaisesRegex(ValueError, "outside platform DRAM"):
            exporter.platform_placement(args, launches)

    def test_missing_invalid_or_excluding_platform_range_rejected(self):
        cases = [(0x80000000, None), (0x80000000, 0), (-64, 1024), (1, 0x80000000),
                 (0x80000000, 1), (0x80000000, True), (0, 1 << 65),
                 (exporter.ARENA_BASE + 64, 1024), (0x80000000, 0x10000000)]
        for base, size in cases:
            with self.subTest(base=base, size=size):
                with self.assertRaises(ValueError):
                    exporter.platform_placement(argparse.Namespace(dram_base=base, dram_bytes=size))


if __name__ == "__main__":
    unittest.main()
