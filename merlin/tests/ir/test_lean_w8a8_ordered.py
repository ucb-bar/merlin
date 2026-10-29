"""Actual-pass ordered bounds, separate numerical demands and compiler-valid mutations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lean_w8a8_ordered_cases import source
from test_lean_w8a8_body import exported
from test_lean_w8a8_receipts import stage_at

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import ordered_evidence, ordered_policy
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def prepare(work, tools, *, before=None, cap=127, index_bits=64, changed=True):
    before = source() if before is None else before
    if changed:
        stage = stage_at(work, before)
    else:
        stage = work / "stage"
        stage.mkdir()
        for name in ("before", "after"):
            (stage / f"{name}.mlir").write_text(str(before))
        (stage / "meta.json").write_text(json.dumps({"boundary": "prepare-quantize-int8"}))
    authoritative = work / "authoritative"
    authoritative.mkdir()
    original = json.loads(exported(before, authoritative, tools, generic=True, allow_unregistered=True).read_text())
    block = original["module"]["ops"][0]["regions"][0]["blocks"][0]
    contraction = next(
        op for op in block["ops"] if op["name"] in {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
    )
    demanded = contraction["results"][0]["id"]
    integers = work / "integers.json"
    integers.write_text(json.dumps([{"function": "forward", "intervals": [None, None]}]))
    exponents = work / "exponents.json"
    exponents.write_text(
        json.dumps(
            [
                {
                    "function": "forward",
                    "exponent_caps": [cap, cap],
                    "source_required_bounds": [{"value": demanded, "upper": 254}],
                    "reference_required_bounds": [],
                }
            ]
        )
    )
    return stage, integers, exponents, index_bits


def invoke(prepared, output, tools):
    stage, integers, exponents, width = prepared
    receipt = ordered_policy.check_w8a8_ordered_draft(
        stage,
        output,
        input_contracts=integers,
        exponent_contracts=exponents,
        m2m_python=tools.m2m_python,
        index_bits=width,
        scope="all",
    )
    assert receipt["verdict"] == "abstracted", receipt
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["runtime_sha256"] == receipt["lean"]["checker_sha256"]
    assert set(report["proof_axioms"]) == set(ordered_evidence.ROOTS)
    assert all(set(value) <= {"propext", "Classical.choice", "Quot.sound"} for value in report["proof_axioms"].values())
    for flag in (
        "whole_pass_verified",
        "policy_reviewed",
        "primitive_model_laws_verified",
        "concrete_context_inhabited",
        "native_primitive_fidelity_proved",
        "transcendental_bounds_assumed",
    ):
        assert report["evidence"][flag] is False
    return receipt, report


@pytest.mark.parametrize("kind", ["matmul", "generic"])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_pass_source_bound_uses_maps_and_reduction_count(tmp_path, lean_tools, kind, shape, index_bits):
    prepared = prepare(
        tmp_path, lean_tools, before=source(kind=kind, m=shape[0], k=shape[1], n=shape[2]), index_bits=index_bits
    )
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]
    function = report["evidence"]["functions"][0]
    prefix = function["source_prefix"]
    demand = prefix["required_bounds"][0]
    fact = next(item for item in prefix["tracked_facts"] if item["id"] == demand["value"])
    assert fact["class"] == "finite"
    assert fact["exponent_upper"] == (0 if shape[1] == 0 else 132)
    assert (prepared[0] / "before.mlir").read_bytes() != (prepared[0] / "after.mlir").read_bytes()


def test_accumulator_first_generic_body_is_checked(tmp_path, lean_tools):
    prepared = prepare(tmp_path, lean_tools, before=source(kind="generic", accumulator_first=True))
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_named_batch_source_bound_is_an_identity_boundary_not_pass_coverage(tmp_path, lean_tools, index_bits):
    # The production W8A8 matcher currently selects matmul and generic contractions,
    # so this tests interpretation of the named source op without claiming a changed site.
    prepared = prepare(tmp_path, lean_tools, before=source(kind="batch_matmul"), index_bits=index_bits, changed=False)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]
    assert (prepared[0] / "before.mlir").read_bytes() == (prepared[0] / "after.mlir").read_bytes()


@pytest.mark.parametrize("parameters,cap", [({}, None), ({}, 254), ({"initial": 1.0}, 127)])
def test_unsupported_numerics_or_initializer_cannot_publish_bound(tmp_path, lean_tools, parameters, cap):
    prepared = prepare(tmp_path, lean_tools, before=source(**parameters), cap=cap)
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert not report["requested_prefix_checks_passed"]
    assert not report["evidence"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]


@pytest.mark.parametrize("mutation", ["body_ssa", "fastmath", "output_map"])
def test_actual_region_and_output_map_are_not_replaced_by_named_templates(tmp_path, lean_tools, mutation):
    prepared = prepare(tmp_path, lean_tools, before=source(kind="generic", m=2, n=2))
    stage = prepared[0]
    text = (stage / "before.mlir").read_text()
    old, new = {
        "body_ssa": ("arith.mulf %x, %y", "arith.mulf %x, %x"),
        "fastmath": ("arith.mulf %x, %y", "arith.mulf %x, %y fastmath<nnan>"),
        "output_map": ("(d0, d1, d2) -> (d0, d1)", "(d0, d1, d2) -> (d1, d0)"),
    }[mutation]
    assert old in text
    changed = parse_mlir_text(text.replace(old, new, 1))
    directory = tmp_path / "valid-source-mutation"
    directory.mkdir()
    exported(changed, directory, lean_tools, generic=True, allow_unregistered=True)
    (stage / "before.mlir").write_text(str(changed))
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert not report["requested_prefix_checks_passed"]
    assert not report["evidence"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]


def test_mutated_target_does_not_hide_passing_source_bounds(tmp_path, lean_tools):
    prepared = prepare(tmp_path, lean_tools)
    stage = prepared[0]
    text = (stage / "after.mlir").read_text()
    old = "math.roundeven"
    assert old in text
    changed = parse_mlir_text(text.replace(old, "math.round", 1))
    directory = tmp_path / "valid-target-mutation"
    directory.mkdir()
    exported(changed, directory, lean_tools, generic=True, allow_unregistered=True)
    (stage / "after.mlir").write_text(str(changed))
    _, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["evidence"]["functions"][0]["source_prefix"]["all_required_bounds_checked"]
    assert not report["evidence"]["structural_match"]
    assert not report["requested_prefix_checks_passed"]


@pytest.mark.parametrize("forgery", [None, "aggregate", "boolean", "source_reference"])
def test_fresh_replay_checks_actual_scoped_evidence(tmp_path, lean_tools, forgery):
    prepared = prepare(tmp_path, lean_tools)
    receipt, report = invoke(prepared, tmp_path / "receipt", lean_tools)
    assert report["requested_prefix_checks_passed"]
    if forgery is not None:
        forged = copy.deepcopy(report)
        if forgery == "aggregate":
            forged["requested_prefix_checks_passed"] = False
        elif forgery == "boolean":
            forged["evidence"]["whole_pass_verified"] = 0
        else:
            item = forged["evidence"]["functions"][0]
            item["source_prefix"], item["reference_prefix"] = item["reference_prefix"], item["source_prefix"]
        path = Path(receipt["cert"]["path"])
        path.write_text(json.dumps(forged))
        receipt["cert"] = _file_ref(path)
        (tmp_path / "receipt/receipt.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="replay differs"):
            ordered_policy.replay(
                tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python
            )
    else:
        replayed = ordered_policy.replay(
            tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python
        )
        fresh = json.loads(Path(replayed["cert"]["path"]).read_text())
        assert fresh["evidence"] == report["evidence"]
