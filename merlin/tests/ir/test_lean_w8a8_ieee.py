"""Actual explicit-model specialization, nonvacuity, valid mutations and hostile evidence."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_receipts import stage_at
from test_lean_w8a8_rotary import prepare

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import ieee_evidence, ieee_policy, ieee_runtime, interpreted
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def invoke(prepared, output, tools):
    stage, integers, caps, width = prepared
    receipt = ieee_policy.check_w8a8_ieee_draft(
        stage,
        output,
        input_contracts=integers,
        exponent_contracts=caps,
        m2m_python=tools.m2m_python,
        index_bits=width,
        scope="captured",
    )
    assert receipt["verdict"] == "abstracted", receipt
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["runtime_sha256"] == receipt["lean"]["checker_sha256"]
    assert set(report["proof_axioms"]) == set(ieee_evidence.ROOTS)
    assert all(set(value) <= {"propext", "Classical.choice", "Quot.sound"} for value in report["proof_axioms"].values())
    evidence = report["evidence"]
    assert evidence["logical_context_inhabitation_proved"] is True
    assert evidence["source_laws_proved"] == 7 and evidence["dynamic_laws_proved"] == 3
    assert evidence["arithmetic_agreement_fields_proved"] == 9
    assert evidence["additional_numerical_hypotheses"] == 0
    for key in (
        "whole_pass_verified",
        "policy_reviewed",
        "primitive_model_laws_verified",
        "concrete_context_inhabited",
    ):
        assert evidence[key] is False
    assert evidence["native_primitive_fidelity_proved"] is False
    assert evidence["frontend_composite_identified"] is False
    return receipt, report


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("cap", [250, 249])
def test_actual_rotary_boundary_specializes_proved_model(tmp_path, lean_tools, index_bits, cap):
    prepared = prepare(tmp_path, lean_tools, index_bits=index_bits, cap=cap)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]
    assert report["evidence"]["joint_input_domain_checked"]


def prepare_boundary(work, tools, before, index_bits, *, dequant_bound=False):
    stage = stage_at(work, before)
    before_dir = work / "before"
    before_dir.mkdir()
    original = json.loads(exported(before, before_dir, tools, generic=True, allow_unregistered=True).read_text())
    block = original["module"]["ops"][0]["regions"][0]["blocks"][0]
    args = block["args"]
    integers = work / "integers.json"
    integers.write_text(json.dumps([{"function": "forward", "intervals": [None] * len(args)}]))
    exponents = work / "exponents.json"
    floating = [arg for arg in args if arg["type"]["kind"] == "tensor" and arg["type"]["element"]["kind"] == "float"]
    caps = [254 if arg in floating else None for arg in args]
    if dequant_bound:
        producer = next(op for op in block["ops"] if op["name"] == "quant_ext.dequantize_per_tensor")
        demanded = producer["results"][0]["id"]
    else:
        assert floating
        demanded = floating[0]["id"]
    exponents.write_text(
        json.dumps(
            [
                {
                    "function": "forward",
                    "exponent_caps": caps,
                    "required_bounds": [{"value": demanded, "upper": 254}],
                }
            ]
        )
    )
    return stage, integers, exponents, index_bits


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("calibration", ["type_attribute", "runtime_scales", "literal_scales"])
def test_complete_actual_boundary_uses_same_proved_context(tmp_path, lean_tools, index_bits, calibration):
    # Give both operands captured dequantize producers, making captured scope
    # select the actual contraction without widening the production selection.
    from test_lean_w8a8_module import calibrated_source

    before = calibrated_source()
    literal_scales = calibration == "literal_scales"
    if calibration != "type_attribute":
        # The declared source adapter accepts string dtype metadata. Keep the original
        # TypeAttr spelling as its own refusal; do not silently change that contract.
        text = str(before)
        assert text.count("input_dtype = i8") == 2
        before = parse_mlir_text(text.replace("input_dtype = i8", 'input_dtype = "int8"'))
    if literal_scales:
        text = str(before)
        arguments = ", %as: tensor<f32>, %bs: tensor<f32>"
        assert text.count(arguments) == 1
        assert text.count("%zp =") == 1
        before = parse_mlir_text(
            text.replace(arguments, "", 1).replace(
                "%zp =",
                "%as_value = arith.constant 0.01 : f32\n"
                "%as = tensor.splat %as_value : tensor<f32>\n"
                "%bs_value = arith.constant 0.02 : f32\n"
                "%bs = tensor.splat %bs_value : tensor<f32>\n%zp =",
                1,
            )
        )
    prepared = prepare_boundary(tmp_path, lean_tools, before, index_bits, dequant_bound=literal_scales)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]
    # Reusing existing byte tensors needs no new quantization of the dequantized data;
    # floating dequantization may return infinity/NaN without producing integer poison.
    assert report["evidence"]["complete_boundary_checked"] is (calibration != "type_attribute")


@pytest.mark.parametrize("mutation", [None, "rounding", "floor", "return"])
def test_complete_dynamic_boundary_and_valid_mutations(tmp_path, lean_tools, mutation):
    from lean_w8a8_numerical_cases import normalization_source

    prepared = prepare_boundary(tmp_path, lean_tools, normalization_source(), 64)
    stage = prepared[0]
    if mutation is not None:
        old, new = {
            "rounding": ("math.roundeven", "math.round"),
            "floor": ("1.000000e-05", "2.000000e-05"),
            "return": ("func.return %result", "func.return %filled"),
        }[mutation]
        text = (stage / "after.mlir").read_text()
        assert old in text
        changed = parse_mlir_text(text.replace(old, new, 1))
        valid = tmp_path / "valid-mutation"
        valid.mkdir()
        exported(changed, valid, lean_tools, generic=True, allow_unregistered=True)
        (stage / "after.mlir").write_text(str(changed))
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_boundary_checked"] is (mutation is None)
    assert report["requested_prefix_checks_passed"] is (mutation is None)
    assert report["evidence"]["generic_rotary"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]


def test_integrity_diagnostic_retains_first_mismatch_without_accepting_a_retry(tmp_path, monkeypatch):
    artifact = tmp_path / "proof.olean"
    artifact.write_bytes(b"qualified")
    expected = _file_ref(artifact)
    runtime = ieee_runtime.Runtime(
        executable=artifact,
        auditor=artifact,
        entry=artifact,
        sysroot=tmp_path,
        search=(),
        modules=(),
        records=(expected,),
        diagnostics=tmp_path / "diagnostics",
    )
    original = ieee_runtime._file_ref
    observations = []

    def changed_once(path):
        observations.append(path)
        if len(observations) == 1:
            return {**expected, "sha256": "0" * 64}
        return original(path)

    monkeypatch.setattr(ieee_runtime, "_file_ref", changed_once)
    with pytest.raises(ValueError, match="first mismatch retained"):
        runtime.assert_current()
    assert observations == [artifact]
    diagnostic = json.loads(next(runtime.diagnostics.glob("mismatch-*.json")).read_text())
    assert diagnostic["expected"] == expected
    assert diagnostic["actual"]["sha256"] == "0" * 64
    assert diagnostic["stat_before"]["bytes"] == len(b"qualified")
    assert "diagnostics" not in runtime.evidence


def test_integrity_diagnostic_records_actual_changed_bytes(tmp_path):
    artifact = tmp_path / "proof.olean"
    artifact.write_bytes(b"qualified")
    expected = _file_ref(artifact)
    runtime = ieee_runtime.Runtime(
        artifact, artifact, artifact, tmp_path, (), (), (expected,), tmp_path / "diagnostics"
    )
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="first mismatch retained"):
        runtime.assert_current()
    diagnostic = json.loads(next(runtime.diagnostics.glob("mismatch-*.json")).read_text())
    assert diagnostic["expected"] == expected
    assert diagnostic["actual"] == _file_ref(artifact)


@pytest.mark.parametrize("cap", [None, 254])
def test_context_witness_does_not_approve_unproved_input_bounds(tmp_path, lean_tools, cap):
    prepared = prepare(tmp_path, lean_tools, cap=cap)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["evidence"]["logical_context_inhabitation_proved"]
    assert not report["requested_prefix_checks_passed"]


@pytest.mark.parametrize("mutation", ["rounding", "floor", "return"])
def test_compiler_valid_changed_target_refuses_under_explicit_model(tmp_path, lean_tools, mutation):
    prepared = prepare(tmp_path, lean_tools)
    stage = prepared[0]
    text = (stage / "after.mlir").read_text()
    old, new = {
        "rounding": ("math.roundeven", "math.round"),
        "floor": ("1.000000e-05", "2.000000e-05"),
        "return": ("func.return %result", "func.return %filled"),
    }[mutation]
    assert old in text
    target = parse_mlir_text(text.replace(old, new, 1))
    valid = tmp_path / "valid-mutation"
    valid.mkdir()
    exported(target, valid, lean_tools, generic=True, allow_unregistered=True)
    (stage / "after.mlir").write_text(str(target))
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert not report["requested_prefix_checks_passed"]
    assert not report["evidence"]["complete_boundary_checked"]
    assert report["evidence"]["generic_rotary"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]


def test_actual_explicit_model_receipt_replays(tmp_path, lean_tools):
    prepared = prepare(tmp_path, lean_tools)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    replayed = ieee_policy.replay(
        tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python
    )
    assert json.loads(Path(replayed["cert"]["path"]).read_text())["evidence"] == report["evidence"]


@pytest.mark.parametrize("forgery", ["native", "premise_type", "context", "green"])
def test_rehashed_report_cannot_replace_model_proofs(tmp_path, lean_tools, forgery):
    prepared = prepare(tmp_path, lean_tools, cap=None)
    receipt, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert not report["requested_prefix_checks_passed"]
    forged = copy.deepcopy(report)
    if forgery == "native":
        forged["evidence"]["native_primitive_fidelity_proved"] = True
    elif forgery == "premise_type":
        forged["evidence"]["logical_context_inhabitation_proved"] = 1
    elif forgery == "context":
        forged["evidence"]["logical_model"] = "invented_model"
    else:
        forged["requested_prefix_checks_passed"] = True
    path = Path(receipt["cert"]["path"])
    path.write_text(json.dumps(forged))
    receipt["cert"] = _file_ref(path)
    original = tmp_path / "receipt/receipt.json"
    original.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="replay differs"):
        ieee_policy.replay(original, tmp_path / "replay", m2m_python=lean_tools.m2m_python)


def test_forged_driver_cannot_change_proved_slice_into_native_admission(tmp_path, lean_tools, monkeypatch):
    prepared = prepare(tmp_path, lean_tools)
    original = interpreted.Runtime.run

    def corrupt(self, arguments, *, timeout_seconds):
        process = original(self, arguments, timeout_seconds=timeout_seconds)
        if self.entry.name == "W8A8IEEEMain.lean":
            evidence = json.loads(process.stdout)
            evidence["native_primitive_fidelity_proved"] = True
            return subprocess.CompletedProcess(process.args, process.returncode, json.dumps(evidence), process.stderr)
        return process

    monkeypatch.setattr(interpreted.Runtime, "run", corrupt)
    stage, integers, caps, width = prepared
    result = ieee_policy.check_w8a8_ieee_draft(
        stage,
        tmp_path / "receipt",
        input_contracts=integers,
        exponent_contracts=caps,
        m2m_python=lean_tools.m2m_python,
        index_bits=width,
    )
    assert result["verdict"] == "unmeasured"
    assert "unproved scope" in result["reason"]


def test_qualification_preserves_existing_receipt_and_contract_alias(tmp_path):
    output = tmp_path / "receipt"
    output.mkdir()
    receipt = output / "receipt.json"
    receipt.write_text("retained evidence")
    with pytest.raises(ValueError, match="new destination"):
        ieee_policy.check_w8a8_ieee_draft(
            tmp_path,
            output,
            input_contracts=receipt,
            exponent_contracts=tmp_path / "caps",
            m2m_python=Path("unused"),
        )
    assert receipt.read_text() == "retained evidence"
