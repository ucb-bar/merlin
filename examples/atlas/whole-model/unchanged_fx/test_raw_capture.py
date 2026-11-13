"""Synthetic raw-intake tests; no real model execution or capture."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import raw_capture as capture


class TinyArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.empty(2, dtype=torch.bfloat16, device="meta"))
        self.proj = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16, device="meta")


def fixture():
    tensors = {"gain": torch.tensor([1, 2], dtype=torch.bfloat16),
               "proj.weight_codes": torch.full((2, 3), 0x38, dtype=torch.uint8),
               "proj.weight_scale_exp": torch.tensor([[0], [1]], dtype=torch.int32),
               "proj.input_scale_exp": torch.tensor(-8, dtype=torch.int32)}
    policy = dict(exported_mode="ref_conv", aliases={}, sites={"proj": dict(
        kind="linear", status="quantized", weight_shape=[2, 3], weight_dtype="bfloat16", bias=None,
        weight=dict(codes="proj.weight_codes", scale_exp="proj.weight_scale_exp", block=0,
                    granularity="per_channel", scale_exp_shape=[2, 1]),
        activation=dict(mode="static", exp=-8, label="static"), residual_passes=None)})
    return tensors, policy


class RawCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tensors, self.policy = fixture()

    def reconstruct(self, model=None, policy=None, tensors=None):
        import load_quantized
        from ref_profile import RefQuantLinear
        tensors = self.tensors if tensors is None else tensors
        policy = self.policy if policy is None else policy
        replay = load_quantized.PolicyAlgo(policy, tensors)
        with capture.prohibit_weight_conversion(), patch.object(
                RefQuantLinear, "__init__", side_effect=AssertionError("quantizing constructor called")):
            return capture.reconstruct(model or TinyArchitecture(), policy, tensors, replay=replay,
                linear_class=RefQuantLinear, patch_attention=lambda model: [])

    def test_raw_constructor_copies_payload_without_cast_or_alias(self):
        before = capture.inventory(self.tensors)
        model = self.reconstruct()
        self.assertEqual(capture.inventory(self.tensors), before)
        self.assertEqual(capture.tensor_record(model.gain), before["gain"])
        self.assertEqual(capture.tensor_record(model.proj.weight_codes), before["proj.weight_codes"])
        self.assertEqual(capture.tensor_record(model.proj.weight_scale_exp), before["proj.weight_scale_exp"])
        self.assertEqual(model.proj.weight.device.type, "meta")
        self.assertFalse(model.training)
        model.proj.weight_codes[0, 0] = 0
        self.assertEqual(self.tensors["proj.weight_codes"][0, 0].item(), 0x38)

    def test_source_linear_eager_forward_uses_raw_buffers(self):
        model = self.reconstruct()
        with capture.prohibit_weight_conversion(), torch.no_grad():
            output = model.proj(torch.ones((1, 3), dtype=torch.bfloat16))
        self.assertTrue(torch.equal(output, torch.tensor([[3, 6]], dtype=torch.bfloat16)))

    def test_tripwires_raise_before_weight_conversion(self):
        import algos.base
        import load_quantized
        import pack_fold
        import quant_modes
        with capture.prohibit_weight_conversion():
            for function in (algos.base.weight_cast, quant_modes.quantize, load_quantized.apply_export,
                             pack_fold.apply_folds, pack_fold.fold_export_dir):
                with self.subTest(function=function), self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function()

    def test_unknown_raw_tensor_is_refused(self):
        self.tensors["unexplained"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "roster"):
            self.reconstruct()

    def test_missing_ordinary_tensor_is_refused(self):
        del self.tensors["gain"]
        with self.assertRaisesRegex(ValueError, "roster"):
            self.reconstruct()

    def test_ordinary_dtype_change_is_refused(self):
        self.tensors["gain"] = self.tensors["gain"].float()
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            self.reconstruct()

    def test_architecture_shape_change_is_refused(self):
        model = TinyArchitecture()
        model.proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "linear mismatch"):
            self.reconstruct(model)

    def test_unsupported_policy_features_fail_closed(self):
        changes = [dict(kind="conv2d"), dict(residual_passes={"kind": "w"}),
                   dict(pv_pop_shift=1), dict(weight_dtype="float32"), dict(bias="bias"),
                   dict(activation=dict(mode="static_per_step", exps=[0, 1]))]
        for change in changes:
            with self.subTest(change=change):
                policy = deepcopy(self.policy)
                policy["sites"]["proj"].update(change)
                with self.assertRaises(ValueError):
                    capture.validate_policy(policy, self.tensors)
        for change in [dict(aliases={"gain": "other"}), dict(pack_fold_applied=True), dict(exported_mode="chip")]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                capture.validate_policy({**self.policy, **change}, self.tensors)

    def test_block_scale_and_wrong_carriers_are_refused(self):
        policy = deepcopy(self.policy)
        policy["sites"]["proj"]["weight"]["block"] = 32
        with self.assertRaisesRegex(ValueError, "block"):
            capture.validate_policy(policy, self.tensors)
        for name, dtype in [("proj.weight_codes", torch.float8_e4m3fn),
                            ("proj.weight_scale_exp", torch.int64)]:
            tensors = dict(self.tensors)
            tensors[name] = tensors[name].view(dtype) if dtype == torch.float8_e4m3fn else tensors[name].to(dtype)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "dtype"):
                capture.validate_policy(self.policy, tensors)

    def test_attention_roster_must_be_authored(self):
        policy = deepcopy(self.policy)
        policy["sites"]["attention"] = dict(kind="attention", status="quantized", variant="text",
            operands={name: dict(mode="dynamic", source="reference") for name in ("q", "k", "p", "v")})
        with self.assertRaisesRegex(ValueError, "attention policy roster"):
            self.reconstruct(policy=policy)

    def source_program(self):
        return SimpleNamespace(named_buffers=lambda: (), state_dict={
            "model.gain": self.tensors["gain"].clone(),
            "model.proj.weight": torch.empty((2, 3), dtype=torch.bfloat16, device="meta"),
            "model.proj.weight_codes": self.tensors["proj.weight_codes"].clone(),
            "model.proj.weight_scale_exp": self.tensors["proj.weight_scale_exp"].clone()})

    def test_original_state_authentication_includes_bf16_and_exponents(self):
        program = self.source_program()
        self.assertEqual(len(capture.authenticate_state(program, self.tensors, self.policy)), 3)
        for name in ("model.gain", "model.proj.weight_codes", "model.proj.weight_scale_exp"):
            with self.subTest(name=name):
                program = self.source_program()
                program.state_dict[name].reshape(-1)[0] += 1
                with self.assertRaisesRegex(ValueError, "differs"):
                    capture.authenticate_state(program, self.tensors, self.policy)

    def test_missing_source_payload_and_unexplained_meta_are_refused(self):
        program = self.source_program()
        del program.state_dict["model.gain"]
        with self.assertRaisesRegex(ValueError, "omits or adds"):
            capture.authenticate_state(program, self.tensors, self.policy)
        program = self.source_program()
        program.state_dict["model.gain"] = torch.empty(2, device="meta")
        with self.assertRaisesRegex(ValueError, "meta placeholder"):
            capture.authenticate_state(program, self.tensors, self.policy)

    def test_lifted_nonpersistent_buffers_are_authenticated(self):
        program = self.source_program()
        buffers = {name: program.state_dict.pop(name) for name in (
            "model.proj.weight_codes", "model.proj.weight_scale_exp")}
        program.named_buffers = lambda: buffers.items()
        self.assertEqual(len(capture.authenticate_state(program, self.tensors, self.policy)), 3)
        buffers["model.proj.weight_scale_exp"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "differs"):
            capture.authenticate_state(program, self.tensors, self.policy)

    def test_tensor_records_hash_raw_words_and_preserve_signed_zero(self):
        values = torch.tensor([0, -32768], dtype=torch.int16).view(torch.bfloat16)
        self.assertNotEqual(capture.tensor_record(values[:1])["sha256"], capture.tensor_record(values[1:])["sha256"])
        self.assertEqual(capture.tensor_record(values)["bytes"], 4)

    def test_manifest_paths_and_file_rosters_are_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "data").write_bytes(b"authenticated")
            record = {"sha256": capture.file_hash(path / "data")}
            capture.verify_files(path, {"data": record}, ("data",))
            with self.assertRaisesRegex(ValueError, "roster"):
                capture.verify_files(path, {"data": record, "extra": record}, ("data",))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                capture.verify_files(path, {"../data": record}, ("../data",))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                capture.verify_files(path, {"data": {"sha256": "wrong"}}, ("data",))

    def test_symlinks_are_rejected_before_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "actual").mkdir()
            (path / "link").symlink_to(path / "actual", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                capture.checked_path(path / "link" / "unopened")

    def test_different_length_gate_refuses_intake_or_stale_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            for gate in ({"schema": "atlas.raw_source_capture.v1", "status": "INTAKE_ONLY"},
                         {"schema": "atlas.raw_source_capture.v1", "status": "SOURCE_PARITY_CAPTURED",
                          "identity": {"changed": True}, "seqlen": 64, "source_shape_parity_cases": 2}):
                path.write_text(__import__("json").dumps(gate))
                with self.assertRaisesRegex(ValueError, "parity gate"):
                    capture.verify_source_gate(path, {}, 64)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
