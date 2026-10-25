"""Actual-pass prefix bounds, compiler-valid mutations, and evidence replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lean_w8a8_rotary_cases import source
from test_lean_w8a8_body import exported
from test_lean_w8a8_receipts import stage_at

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import rotary_evidence, rotary_policy
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def prepare(work, tools, *, before=None, cap=250, index_bits=64):
    work.mkdir(exist_ok=True)
    before = source() if before is None else before
    stage = stage_at(work, before)
    export_dir = work / "authoritative"
    export_dir.mkdir()
    original = json.loads(exported(before, export_dir, tools, generic=True, allow_unregistered=True).read_text())
    ops = original["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    selected = [op for op in ops if op["name"] in {"linalg.transpose", "tensor.concat"} or "prov.bound" in op["attrs"]]
    demands = [{"value": op["results"][0]["id"], "upper": 254} for op in selected]
    assert len(demands) == 3
    integers = work / "integers.json"
    integers.write_text(json.dumps([{"function": "forward", "intervals": [None, None]}]))
    exponents = work / "exponents.json"
    exponents.write_text(
        json.dumps([{"function": "forward", "exponent_caps": [cap, None], "required_bounds": demands}])
    )
    return stage, integers, exponents, index_bits


def invoke(prepared, work, tools):
    stage, integers, exponents, index_bits = prepared
    return rotary_policy.check_w8a8_rotary_draft(
        stage,
        work,
        input_contracts=integers,
        exponent_contracts=exponents,
        m2m_python=tools.m2m_python,
        scope="captured",
        index_bits=index_bits,
    )


def evidence(receipt):
    assert receipt["verdict"] == "abstracted", receipt
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["runtime_sha256"] == receipt["lean"]["checker_sha256"]
    assert set(report["proof_axioms"]) == set(rotary_evidence.ROOTS)
    assert all(
        set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in report["proof_axioms"].values()
    )
    assert report["evidence"]["whole_pass_verified"] is False
    assert report["evidence"]["primitive_model_laws_verified"] is False
    return report


@pytest.mark.parametrize("rows,channels", [(3, 2), (8, 3), (5, 7), (0, 2)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_changed_pass_keeps_checked_rotary_bounds(tmp_path, lean_tools, rows, channels, index_bits):
    prepared = prepare(tmp_path, lean_tools, before=source(rows=rows, channels=channels), index_bits=index_bits)
    report = evidence(invoke(prepared, tmp_path / "receipt", lean_tools))
    assert report["requested_prefix_checks_passed"]
    assert report["evidence"]["structural_match"]
    assert (prepared[0] / "before.mlir").read_text() != (prepared[0] / "after.mlir").read_text()
    function = report["evidence"]["functions"][0]
    assert function["domain_inhabited"]
    assert all(function[name]["all_required_bounds_checked"] for name in ("source_prefix", "reference_prefix"))


@pytest.mark.parametrize(
    "parameters,cap",
    [({}, None), ({}, 254), ({"rows": 16}, 250), ({"step": 512}, 250), ({"arithmetic": "arith.divf"}, 250)],
)
def test_unjustified_bounds_refuse_after_authoritative_export(tmp_path, lean_tools, parameters, cap):
    prepared = prepare(tmp_path, lean_tools, before=source(**parameters), cap=cap)
    report = evidence(invoke(prepared, tmp_path / "receipt", lean_tools))
    assert report["evidence"]["structural_match"]
    assert not report["requested_prefix_checks_passed"]
    assert not report["evidence"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]


@pytest.mark.parametrize("mutation", ["rounding", "constant", "return", "transpose", "angle_ssa"])
def test_valid_changed_targets_refuse_even_when_source_bounds_pass(tmp_path, lean_tools, mutation):
    prepared = prepare(tmp_path, lean_tools, before=source(rows=3, channels=3))
    stage = prepared[0]
    text = (stage / "after.mlir").read_text()
    replacements = {
        "rounding": ("math.roundeven", "math.round"),
        "constant": ("1.000000e-05", "2.000000e-05"),
        "return": ("func.return %result", "func.return %filled"),
        "transpose": ("permutation = [1, 0]", "permutation = [0, 1]"),
        "angle_ssa": ("arith.mulf %angle_frequency, %angle_position", "arith.mulf %angle_frequency, %angle_initial"),
    }
    old, new = replacements[mutation]
    assert old in text
    changed = parse_mlir_text(text.replace(old, new, 1))
    directory = tmp_path / "valid-mutation"
    directory.mkdir()
    exported(changed, directory, lean_tools, generic=True, allow_unregistered=True)
    (stage / "after.mlir").write_text(str(changed))
    report = evidence(invoke(prepared, tmp_path / "receipt", lean_tools))
    assert not report["evidence"]["structural_match"]
    assert report["evidence"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]
    assert not report["requested_prefix_checks_passed"]


def test_receipt_replays_the_actual_runtime_and_source_bounds(tmp_path, lean_tools):
    prepared = prepare(tmp_path, lean_tools)
    receipt = invoke(prepared, tmp_path / "receipt", lean_tools)
    report = evidence(receipt)
    replayed = rotary_policy.replay(
        tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python
    )
    assert evidence(replayed)["evidence"] == report["evidence"]


@pytest.mark.parametrize("forgery", ["green", "premise_type", "prefix_count", "duplicate_premise"])
def test_forged_report_cannot_certify_its_own_replay(tmp_path, lean_tools, forgery):
    prepared = prepare(tmp_path, lean_tools, cap=None)
    receipt = invoke(prepared, tmp_path / "receipt", lean_tools)
    report = evidence(receipt)
    assert not report["requested_prefix_checks_passed"]
    forged = copy.deepcopy(report)
    if forgery == "green":
        forged["requested_prefix_checks_passed"] = True
    elif forgery == "premise_type":
        # Ordinary Python equality treats 1 and True as equal, even in nested
        # dictionaries. Rehashing a receipt must not hide a changed premise type.
        forged["evidence"]["arithmetic_agreement_assumed"] = 1
    elif forgery == "prefix_count":
        forged["evidence"]["functions"][0]["source_prefix"]["operation_count"] += 1
    report_path = Path(receipt["cert"]["path"])
    text = json.dumps(forged)
    if forgery == "duplicate_premise":
        old = '"arithmetic_agreement_assumed": true'
        assert old in text
        text = text.replace(old, '"arithmetic_agreement_assumed": false, ' + old, 1)
    report_path.write_text(text)
    receipt["cert"] = _file_ref(report_path)
    receipt_path = tmp_path / "receipt/receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="duplicate JSON key" if forgery == "duplicate_premise" else "replay differs"):
        rotary_policy.replay(receipt_path, tmp_path / "replay", m2m_python=lean_tools.m2m_python)


def test_qualification_and_replay_preserve_existing_receipts_and_contracts(tmp_path):
    destination = tmp_path / "receipt"
    destination.mkdir()
    receipt = destination / "receipt.json"
    receipt.write_text("retained bytes")
    with pytest.raises(ValueError, match="new destination"):
        rotary_policy.check_w8a8_rotary_draft(
            tmp_path,
            destination,
            input_contracts=receipt,
            exponent_contracts=tmp_path / "caps.json",
            m2m_python=Path("/unused"),
        )
    assert receipt.read_text() == "retained bytes"
