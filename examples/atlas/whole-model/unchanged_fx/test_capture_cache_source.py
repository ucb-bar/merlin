"""Synthetic cache-capture admission, source parity, and publication tests."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import capture_cache_source as capture
from cache_source import TensorCacheLogits, TensorPrefillLogits, phase_positions_and_mask


class CaptureCacheSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="cache-source-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        first = torch.arange(64)[None]
        second = first + 1
        return {"first": first, "second": second}, {"cases": [
            {"inputs": ["first"]}, {"inputs": ["second"]}]}

    def phase_cases(self, layer_count=1):
        cases = {}
        for name, past, query in capture.PHASES:
            cases[name] = []
            for index in range(2):
                ids = torch.full((1, query), index + 1, dtype=torch.int64)
                positions, mask = phase_positions_and_mask(ids, past)
                cache = () if past == 0 else tuple(torch.full((1, 1, past, 2), index, dtype=torch.bfloat16)
                                                  for _ in range(2 * layer_count))
                cases[name].append((ids, positions, mask, *cache))
        return cases

    def test_original_fixture_selection_preserves_words_and_provenance(self):
        fixtures, parity = self.fixture()
        sequences, origins = capture.fixture_sequences(fixtures, parity)
        self.assertTrue(torch.equal(sequences[0], fixtures["first"][:, :6]))
        self.assertTrue(torch.equal(sequences[1], fixtures["second"][:, :6]))
        self.assertEqual(origins[0]["source_input"], "first")
        sequences[0][0, 0] = 31
        self.assertEqual(fixtures["first"][0, 0].item(), 0)

    def test_invalid_original_fixtures_are_refused(self):
        fixtures, parity = self.fixture()
        for invalid in (torch.ones((1, 5), dtype=torch.int64), torch.ones((1, 64)),
                        torch.ones((2, 64), dtype=torch.int64)):
            with self.subTest(shape=invalid.shape), self.assertRaisesRegex(ValueError, "fixture needs"):
                capture.fixture_sequences({**fixtures, "first": invalid}, parity)
        with self.assertRaisesRegex(ValueError, "prefixes must differ"):
            capture.fixture_sequences({"first": fixtures["first"], "second": fixtures["first"]}, parity)
        with self.assertRaisesRegex(ValueError, "two original"):
            capture.fixture_sequences(fixtures, {"cases": parity["cases"][:1]})

    def test_full_source_interface_names_all_44_cache_tensors_and_88_links(self):
        interface = capture.phase_interface(22)
        self.assertEqual(len(interface["inputs"]), 47)
        self.assertEqual(len(interface["outputs"]), 45)
        self.assertEqual(len(interface["phases"]["prefill4"]["inputs"]), 3)
        self.assertTrue(interface["phases"]["prefill4"]["initial_cache_empty"])
        self.assertEqual(len(interface["phases"]["decode5"]["inputs"]), 47)
        self.assertEqual(len(interface["source_cache_links"]), 88)
        self.assertEqual(interface["inputs"][-1], {"index": 46, "role": "past_value", "layer": 21})
        self.assertEqual(interface["outputs"][-1], {"index": 44, "role": "updated_value", "layer": 21})
        self.assertEqual(len({(row["consumer"], row["input_index"])
                              for row in interface["source_cache_links"]}), 88)

    def arguments(self):
        return ["--bundle", str(self.root / "bundle"), "--export-dir", str(self.root / "export"),
                "--architecture", str(self.root / "architecture"), "--source-shape-gate", str(self.root / "gate.json"),
                "--out", str(self.root / "output")]

    def test_unfinished_raw64_gate_refuses_before_intake_or_construction(self):
        (self.root / "gate.json").write_text(json.dumps({"status": "RUNNING", "seqlen": 64}))
        with patch.object(capture.raw, "source_identity") as intake, patch.object(capture, "construct_source") as construct:
            with self.assertRaisesRegex(ValueError, "completed raw64"):
                capture.main(self.arguments())
            intake.assert_not_called()
            construct.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_stale_completed_gate_refuses_before_construction(self):
        (self.root / "gate.json").write_text(json.dumps({"status": "SOURCE_PARITY_CAPTURED", "seqlen": 64}))
        with patch.object(torch, "set_num_threads"), patch.object(torch, "set_num_interop_threads"), patch.object(
                capture.raw, "source_identity", return_value=({"provenance": {"seqlen": 64}}, {})), patch.object(
                capture.raw, "verify_source_gate", side_effect=ValueError("stale raw gate")), patch.object(
                capture, "construct_source") as construct:
            with self.assertRaisesRegex(ValueError, "stale raw gate"):
                capture.main(self.arguments())
            construct.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def run_synthetic_cli(self, *, parity_only=False, parity_failure=False):
        """Exercise CLI sequencing with mocked model/capture, no real source weights."""
        (self.root / "gate.json").write_text(json.dumps({"status": "SOURCE_PARITY_CAPTURED", "seqlen": 64}))
        bundle = self.root / "bundle"
        bundle.mkdir()
        fixtures, parity = self.fixture()
        (bundle / "parity.json").write_text(json.dumps(parity))
        model = torch.nn.Module().eval()
        model.config = SimpleNamespace(num_hidden_layers=1)
        tensors = {"raw_codes": torch.tensor([0x38], dtype=torch.uint8)}
        frozen = capture.raw.inventory(tensors)
        before = capture.module_inventory(model)
        events = []
        def construct(*args):
            events.append("construct")
            return model, {}, tensors, frozen, {"source": "unchanged"}
        def compare(*args):
            events.append("parity")
            if parity_failure:
                raise ValueError("synthetic source parity mismatch")
            return {name: model for name, _, _ in capture.PHASES}, self.phase_cases(), [{"status": "SOURCE_WORDS_EXACT"}] * 6, before
        def publish(boundary, cases, directory, **kwargs):
            events.append(directory.name)
            self.assertEqual(kwargs["provenance"]["phase"], directory.name)
            self.assertEqual(len(cases), 2)
            return {"status": "SOURCE_CAPTURED"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch, "set_num_threads"))
            stack.enter_context(patch.object(torch, "set_num_interop_threads"))
            stack.enter_context(patch.object(capture.raw, "source_identity", return_value=({
                "provenance": {"seqlen": 64}}, {"synthetic": True})))
            guard = stack.enter_context(patch.object(capture.raw, "verify_source_gate"))
            stack.enter_context(patch.object(capture, "construct_source", side_effect=construct))
            stack.enter_context(patch.object(capture, "collect_parity_cases", side_effect=compare))
            stack.enter_context(patch("safetensors.torch.load_file", return_value=fixtures))
            stack.enter_context(patch.object(capture, "capture_phase", side_effect=publish))
            args = self.arguments() + (["--parity-only"] if parity_only else [])
            if parity_failure:
                with self.assertRaisesRegex(ValueError, "synthetic source parity mismatch"):
                    capture.main(args)
            else:
                self.assertEqual(capture.main(args), 0)
            guard.assert_called_once()
        return events, json.loads((self.root / "output/cache_capture.json").read_text())

    def test_full_cli_publishes_three_phases_only_after_source_parity(self):
        events, receipt = self.run_synthetic_cli()
        self.assertEqual(events, ["construct", "parity", "prefill4", "decode5", "decode6"])
        self.assertEqual(receipt["status"], "SOURCE_CACHE_CAPTURED")
        self.assertEqual(receipt["source_phase_comparisons"], 6)
        self.assertEqual(receipt["generated_execution"], "UNRUN")

    def test_parity_only_cli_never_captures(self):
        events, receipt = self.run_synthetic_cli(parity_only=True)
        self.assertEqual(events, ["construct", "parity"])
        self.assertEqual(receipt["status"], "SOURCE_CACHE_PARITY_ONLY")
        self.assertEqual(receipt["phases"], {})

    def test_source_parity_failure_persists_failed_receipt_without_capture(self):
        events, receipt = self.run_synthetic_cli(parity_failure=True)
        self.assertEqual(events, ["construct", "parity"])
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(receipt["phases"], {})

    def test_two_actual_tiny_source_chains_match_original_forward(self):
        from test_cache_source import tiny_source

        model = tiny_source()
        sequences = [torch.tensor([[2, 3, 5, 7, 11, 13]]), torch.tensor([[17, 19, 23, 29, 31, 3]])]
        boundaries, cases, records, before = capture.collect_parity_cases(model, sequences)
        self.assertIsInstance(boundaries["prefill4"], TensorPrefillLogits)
        self.assertEqual(len(records), 6)
        self.assertEqual(capture.module_inventory(model), before)
        for name, past, query in capture.PHASES:
            self.assertEqual(len(cases[name]), 2)
            self.assertEqual(tuple(cases[name][0][0].shape), (1, query))
            self.assertEqual(len(cases[name][0]), 3 if past == 0 else 7)
            if past:
                self.assertEqual(cases[name][0][3].shape[-2], past)
        for case in range(2):
            rows = [row for row in records if row["case"] == case]
            for previous, current in zip(rows, rows[1:]):
                for cache_index in range(4):
                    self.assertEqual(previous["outputs"][f"output_{1 + cache_index}"]["sha256"],
                                     current["inputs"][f"input_{3 + cache_index}"]["sha256"])

    def test_wrong_wrapper_words_refuse_before_capture(self):
        from test_cache_source import tiny_source

        class Corrupt(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.boundary = TensorPrefillLogits(model)
            def forward(self, *args):
                outputs = self.boundary(*args)
                logits = outputs[0].clone()
                logits.view(torch.int16).reshape(-1)[0] ^= 1
                return (logits, *outputs[1:])
        with patch.object(capture, "TensorPrefillLogits", Corrupt), self.assertRaisesRegex(ValueError, "parity differs"):
            capture.collect_parity_cases(tiny_source(), [torch.tensor([[2, 3, 5, 7, 11, 13]]),
                                                       torch.tensor([[17, 19, 23, 29, 31, 3]])])

    def test_strict_source_native_cache_export_replays(self):
        import fx_export
        from test_cache_source import raw_hash, tiny_source

        model = tiny_source()
        decode, prefill = TensorCacheLogits(model).eval(), TensorPrefillLogits(model).eval()
        replay = model.model.layers[0].self_attn._atlas_ref_attn["algo"]
        tensors = None
        with torch.no_grad():
            for ids in (torch.tensor([[2, 3, 5, 7]]), torch.tensor([[11]]), torch.tensor([[13]])):
                past = 0 if tensors is None else tensors[0].shape[-2]
                positions, mask = phase_positions_and_mask(ids, past)
                boundary = prefill if tensors is None else decode
                args = (ids, positions, mask) if tensors is None else (ids, positions, mask, *tensors)
                expected = boundary(*args)
                with fx_export._native_sites(boundary, replay.policy, "uint8"):
                    program = torch.export.export(boundary, args, strict=True)
                abi = capture.derive_cache_phase_abi(program, layer_count=2,
                    past_length=past, query_length=ids.shape[1], kv_heads=2, head_dim=8, batch_size=1)
                self.assertEqual(len(abi["inputs"]), 3 if past == 0 else 7)
                self.assertEqual(len(abi["outputs"]), 5)
                actual = program.module()(*args)
                self.assertEqual([raw_hash(value) for value in actual], [raw_hash(value) for value in expected])
                tensors = expected[1:]

    def synthetic_capture(self, *, state=None, missing_fresh=False, mutate_inputs=False):
        """Fake publisher exercises post-capture authentication, never numerical capture."""
        fresh = dict(status="passed", all_outputs_byte_identical=True, code_scale_bytes_preserved=True,
                     model_replay_modules_imported=[])
        report = dict(status="passed", fresh_process_requested=not missing_fresh,
                      fresh_process_reload=fresh, native_fp8_detail=dict(storage="uint8",
                          code_bytes_equal_uint8_source=True, outputs_identical_to_decode_graph=True))
        observed = {}
        def publish(boundary, cases, directory, **kwargs):
            observed.update(kwargs)
            directory.mkdir()
            (directory / "model.pt2").write_bytes(b"synthetic export")
            (directory / "verification.json").write_text(json.dumps(report))
            (directory / "fresh_process_verification.json").write_text(json.dumps(fresh))
            (directory / "fx_manifest.json").write_text(json.dumps({"files": {
                "model.pt2": {"sha256": capture.raw.file_hash(directory / "model.pt2")}}}))
            if mutate_inputs:
                cases[0][0][0, 0] = 31
            return report
        stub = SimpleNamespace(capture_fx=publish, BUNDLE_FILES=("model.pt2",))
        program = SimpleNamespace(graph_signature=SimpleNamespace(
            input_specs=[SimpleNamespace(kind=SimpleNamespace(name="USER_INPUT")) for _ in range(3)],
            output_specs=[SimpleNamespace(kind=SimpleNamespace(name="USER_OUTPUT")) for _ in range(3)]))
        original_import = importlib.import_module
        with patch.object(capture.importlib, "import_module", side_effect=lambda name: stub if name == "fx_export"
                          else original_import(name)), patch.object(capture.raw, "prohibit_weight_conversion",
                          side_effect=contextlib.nullcontext), patch.object(torch.export, "load", return_value=program), patch.object(
                          capture.raw, "authenticate_state", return_value=state or {"source": "unchanged"}), patch.object(
                          capture, "derive_cache_phase_abi", return_value={"schema": "synthetic phase metadata", "initial_cache_empty": True}):
            result = capture.capture_phase(None, self.phase_cases()["prefill4"], self.root / "phase",
                policy={}, provenance={}, tensors={}, source_state={"source": "unchanged"}, layer_count=1)
        return result, observed

    def test_capture_demands_native_uint8_fresh_process_and_reauthenticates_state(self):
        result, observed = self.synthetic_capture()
        self.assertEqual(result["status"], "SOURCE_CAPTURED")
        self.assertEqual(result["input_count"], 3)
        self.assertTrue(result["initial_cache_empty"])
        self.assertEqual(result["output_count"], 3)
        self.assertEqual(observed["native_fp8"], "uint8")
        self.assertTrue(observed["fresh_process"])
        binding = json.loads((self.root / "phase/cache_phase_abi.json").read_text())
        self.assertEqual(binding["schema"], "atlas.source_cache_phase_binding.v1")
        self.assertEqual(binding["model_pt2_sha256"], result["model_pt2_sha256"])
        self.assertEqual(binding["source_fx_manifest_sha256"], result["manifest_sha256"])
        self.assertEqual(capture.raw.file_hash(self.root / "phase/cache_phase_abi.json"),
                         result["cache_phase_abi_sha256"])

    def test_capture_refuses_missing_fresh_process_evidence(self):
        with self.assertRaisesRegex(ValueError, "fresh-process qualification"):
            self.synthetic_capture(missing_fresh=True)

    def test_capture_refuses_changed_saved_source_state(self):
        with self.assertRaisesRegex(ValueError, "source state differs"):
            self.synthetic_capture(state={"source": "changed"})

    def test_capture_refuses_mutated_admitted_inputs(self):
        with self.assertRaisesRegex(ValueError, "changed its admitted phase inputs"):
            self.synthetic_capture(mutate_inputs=True)

    def test_capture_rejects_legacy_empty_tensor_prefill_interface(self):
        cases = [(*case, torch.empty((1, 1, 0, 2), dtype=torch.bfloat16),
                         torch.empty((1, 1, 0, 2), dtype=torch.bfloat16))
                 for case in self.phase_cases()["prefill4"]]
        with self.assertRaisesRegex(ValueError, "prefill capture requires three inputs"):
            capture.capture_phase(None, cases, self.root / "refused", policy={}, provenance={},
                                  tensors={}, source_state={}, layer_count=1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
