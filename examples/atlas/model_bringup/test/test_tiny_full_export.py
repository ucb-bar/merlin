"""Controls for exact-byte shared assets and early capture admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("tiny_full_export", Path(__file__).absolute().parents[1]
                                            / "tools/tiny_full/export_capture.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class FullExportTest(unittest.TestCase):
    def dispatch_fixture(self, root, mode=False, calls=None, include_calls=True):
        report = {"status": "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY", "decode_count": 1,
                  "atlas_instruction_execution": mode, "source_records": [],
                  **{name: True for name in ("source_bookends_verified", "target_replay_exact", "greedy_feedback_actual",
                                             "all22_cache_prefixes_exact", "native_restored_exact")}}
        if include_calls:
            report["attention_calls"] = [] if calls is None else calls
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps(report))
        return argparse.Namespace(capture=receipt, capture_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                  model_root=root / "model", out=root / "bundle")

    def test_legacy_absent_and_empty_attention_ledgers_use_legacy_export(self):
        for explicit_mode in (True, False):
            for include_calls in (True, False):
                with self.subTest(explicit_mode=explicit_mode, include_calls=include_calls):
                    with tempfile.TemporaryDirectory() as directory:
                        args = self.dispatch_fixture(Path(directory), include_calls=include_calls)
                        if not explicit_mode:
                            data = json.loads(args.capture.read_text())
                            del data["atlas_instruction_execution"]
                            args.capture.write_text(json.dumps(data))
                            args.capture_sha256 = hashlib.sha256(args.capture.read_bytes()).hexdigest()
                        with mock.patch.object(exporter, "model_binding", side_effect=RuntimeError("legacy export")):
                            with self.assertRaisesRegex(RuntimeError, "legacy export"):
                                exporter.export(args)
                        self.assertFalse(args.out.exists())

    def test_explicit_complete_attention_dispatch(self):
        calls = [{"index": step * 22 + layer, "step": step, "layer": layer}
                 for step in range(2) for layer in range(22)]
        with tempfile.TemporaryDirectory() as directory:
            args = self.dispatch_fixture(Path(directory), mode=True, calls=calls)
            attention = mock.Mock(return_value="attention export")
            with mock.patch.dict("sys.modules", {"attention_export": types.SimpleNamespace(export_attention=attention)}):
                self.assertEqual(exporter.export(args), "attention export")
            attention.assert_called_once_with(args)
            self.assertFalse(args.out.exists())

    def test_contradictory_or_incomplete_attention_rejected_before_model_or_output(self):
        calls = [{"index": step * 22 + layer, "step": step, "layer": layer}
                 for step in range(2) for layer in range(22)]
        cases = [(False, calls, True), (True, [], True), (True, None, False),
                 (True, calls[:-1], True), ("true", calls, True), (False, {}, True),
                 (True, [dict(call, step=0) for call in calls], True)]
        for mode, ledger, include_calls in cases:
            with self.subTest(mode=mode, ledger=ledger, include_calls=include_calls):
                with tempfile.TemporaryDirectory() as directory:
                    args = self.dispatch_fixture(Path(directory), mode=mode, calls=ledger, include_calls=include_calls)
                    with mock.patch.object(exporter, "model_binding") as model:
                        with self.assertRaises(ValueError):
                            exporter.export(args)
                        model.assert_not_called()
                    self.assertFalse(args.out.exists())

    def test_stale_imported_layout_is_refused(self):
        with mock.patch.object(exporter.atlas_host_layout, "HEADER_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "host layout changed"):
                exporter.checked_layout()

    @unittest.skipUnless(os.environ.get("TINY_MODEL_ROOT") and os.environ.get("TINY_FULL_CAPTURE"),
                         "set explicit model and full capture fixtures")
    def test_vector_codec_selected_rtl_boundaries_and_actual_weight(self):
        root = exporter.checked(os.environ["TINY_MODEL_ROOT"])
        _, torch, _, _ = exporter.configure_model(root)
        from npu_model.configs.numerics import RtlNumerics
        capture = exporter.checked(os.environ["TINY_FULL_CAPTURE"])
        report = json.loads(capture.read_text())
        codec_path = [row["path"] for row in report["source_records"]
                      if Path(row["path"]).name == "tiny_fp8_pack.py"][0]
        spec = importlib.util.spec_from_file_location("tiny_codec_test", exporter.checked(codec_path))
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        words = np.arange(65536, dtype=np.uint16)
        words = words[(words & 0x7f80) != 0x7f80]
        for scale in (-126, -6, 0, 7, 127):
            expected = codec.pack_words(words.reshape(1, -1), scale)
            actual = RtlNumerics.to_fp8(torch.from_numpy(words.copy()).view(torch.bfloat16),
                                        127 + scale).view(torch.uint8).numpy()
            self.assertEqual(expected.tobytes(), actual.tobytes())
        descriptor = report["parameters"]["model.layers.0.self_attn.q_proj"]["B_bf16"]
        weight, _ = exporter.read_array(capture.parent, descriptor)
        encoded, scale = exporter.ExactEncoder(codec, torch, RtlNumerics.to_fp8).encode(weight)
        self.assertEqual(encoded.shape, weight.shape)
        self.assertEqual(scale, codec.tensor_scale_bits(weight))

    def test_encoded_prepare_preserves_shared_packing_and_assembly(self):
        a = np.arange(32 * 64, dtype=np.uint8).reshape(32, 64)
        b = np.arange(64 * 32, dtype=np.uint8).reshape(64, 32)
        class Assembled:
            def assemble(self):
                return [0x13, 0x73]
        def assemble(source):
            self.assertIn("vmatmul.acc.mxu0", source.read())
            return Assembled()
        result = exporter.prepare_encoded(a, b, -3, assemble)
        self.assertEqual(result["activation"], exporter.pack_activation(a))
        self.assertEqual(result["weights"], exporter.pack_weights(b))
        self.assertEqual(result["memory"]["output_offset"], 4352)
        self.assertEqual(result["initial"], result["activation"] + result["weights"]
                         + b"\xa5" * 256 + b"\xcd" * 2048 + b"\x5a" * 256)
        self.assertEqual(result["requested_dma"], {"load_bytes": 4096, "store_bytes": 2048})

    def test_identical_bytes_share_asset_distinct_rescale_words_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            store = exporter.Assets(Path(directory))
            first = np.array([0x00000013, 0x3f800037, 0x73], dtype="<u4").tobytes()
            second = np.array([0x00000013, 0x40000037, 0x73], dtype="<u4").tobytes()
            path = store.put("programs", first)
            self.assertEqual(store.put("programs", bytes(first)), path)
            self.assertNotEqual(store.put("programs", second), path)
            self.assertEqual(len(store.sums), 2)

    def test_array_payload_is_bound_before_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.array([[0x3f80, 0x8000]], dtype="<u2").tobytes()
            (root / "words.bin").write_bytes(raw)
            descriptor = {"file": "words.bin", "shape": [1, 2], "dtype": "uint16", "bytes": 4,
                          "sha256": hashlib.sha256(raw).hexdigest()}
            words, _ = exporter.read_array(root, descriptor)
            self.assertEqual(words.tobytes(), raw)
            (root / "words.bin").write_bytes(raw[:-1] + b"\x01")
            with self.assertRaises(ValueError):
                exporter.read_array(root, descriptor)

    def test_partial_capture_refused_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({"status": "failed"}))
            args = argparse.Namespace(capture=receipt, capture_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                      model_root=root / "model", out=root / "bundle")
            with mock.patch.object(exporter, "configure_model") as model:
                with self.assertRaises(ValueError):
                    exporter.export(args)
                model.assert_not_called()
            self.assertFalse(args.out.exists())



if __name__ == "__main__":
    unittest.main()
