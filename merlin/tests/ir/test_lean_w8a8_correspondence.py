"""Complete policy correspondence, independent execution obligations and actual replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lean_w8a8_projection_cases import CASES
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import calibrated_source, source
from test_lean_w8a8_receipts import stage_at
from test_lean_w8a8_views import evaluate, words

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import correspondence_evidence as checks
from merlin.verify.lean import correspondence_policy as policy
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def invoke(stages, destination, tools, *, scope="all", index_bits=64):
    receipt = policy.check_w8a8_correspondence_draft(
        stages, destination, m2m_python=tools.m2m_python, scope=scope, index_bits=index_bits
    )
    assert receipt["verdict"] == "abstracted", receipt
    report = checks.read(Path(receipt["cert"]["path"]))
    assert report["runtime_sha256"] == receipt["lean"]["checker_sha256"]
    assert set(report["proof_axioms"]) == set(checks.ROOTS)
    assert all(
        set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in report["proof_axioms"].values()
    )
    evidence = report["evidence"]
    assert evidence["numerical_premises_for_correspondence"] == 0
    assert evidence["original_source_definedness_required_for_correspondence"] is False
    for key in (
        "whole_pass_verified",
        "policy_reviewed",
        "primitive_model_laws_verified",
        "concrete_context_inhabited",
    ):
        assert evidence[key] is False
    assert evidence["whole_model_initialized_execution_established"] is False
    return receipt, report


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2), (3, 2, 3)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_pass_correspondence_covers_complete_module(tmp_path, lean_tools, shape, index_bits):
    stage = stage_at(tmp_path, source(*shape))
    receipt, report = invoke([stage], tmp_path / "receipt", lean_tools, index_bits=index_bits)
    assert report["evidence"]["complete_correspondence_passed"]
    assert report["evidence"]["actual_boundary_changed"]
    assert receipt["before"]["sha256"] != receipt["after"]["sha256"]


@pytest.mark.parametrize("name", ["interleaved", "two_reductions", "scalar", "attention"])
def test_projection_geometries_use_same_complete_boundary_claim(tmp_path, lean_tools, name):
    stage = stage_at(tmp_path, CASES[name].source(integer_left=name == "interleaved"))
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_correspondence_passed"]


def test_calibrated_byte_source_needs_no_extra_law_for_correspondence(tmp_path, lean_tools):
    stage = stage_at(tmp_path, calibrated_source())
    _, report = invoke([stage], tmp_path / "receipt", lean_tools, scope="captured")
    assert report["evidence"]["complete_correspondence_passed"]


@pytest.mark.parametrize("reduction,accepted", [(131071, True), (131072, False)])
def test_unsupported_selected_contraction_refuses_complete_boundary(tmp_path, lean_tools, reduction, accepted):
    stage = stage_at(tmp_path, source(1, reduction, 1))
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["reference_expanded"] is accepted
    assert report["evidence"]["complete_correspondence_passed"] is accepted


def test_omitting_selected_transformation_is_not_policy_correspondence(tmp_path, lean_tools):
    stage = stage_at(tmp_path, source())
    (stage / "after.mlir").write_text((stage / "before.mlir").read_text())
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["reference_expanded"]
    assert report["evidence"]["complete_correspondence_passed"] is False


def two_outputs():
    return parse_mlir_text("""func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %e = tensor.empty() : tensor<2x2xf32>
      %z = arith.constant 0.0 : f32
      %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
      return %r, %a : tensor<2x2xf32>, tensor<2x2xf32>
    }""")


def valid_replacement(stage, directory, tools, old, new):
    text = (stage / "after.mlir").read_text()
    assert old in text
    target = parse_mlir_text(text.replace(old, new, 1))
    directory.mkdir()
    exported(target, directory, tools, generic=True, allow_unregistered=True)
    (stage / "after.mlir").write_text(str(target))


@pytest.mark.parametrize("mutation", [None, "rounding", "floor", "second_output", "visibility", "declaration"])
def test_every_output_and_outer_change_is_checked(tmp_path, lean_tools, mutation):
    stage = stage_at(tmp_path, two_outputs())
    if mutation:
        text = (stage / "after.mlir").read_text()
        if mutation == "second_output":
            original = next(line.strip() for line in text.splitlines() if "func.return" in line)
            operands = original.split(" : ")[0].removeprefix("func.return ").split(", ")
            assert len(operands) == 2
            changed = original.replace(operands[1], "%b", 1)
            valid_replacement(stage, tmp_path / "mutation", lean_tools, original, changed)
        elif mutation == "declaration":
            # A legal extra declaration must not disappear from complete-module comparison.
            prefix, suffix = text.rsplit("}", 1)
            target = parse_mlir_text(prefix + "func.func private @extra()\n}" + suffix)
            directory = tmp_path / "mutation"
            directory.mkdir()
            exported(target, directory, lean_tools, generic=True, allow_unregistered=True)
            (stage / "after.mlir").write_text(str(target))
        else:
            old, new = {
                "rounding": ("math.roundeven", "math.round"),
                "floor": ("1.000000e-05", "2.000000e-05"),
                "visibility": ("func.func @forward", "func.func private @forward"),
            }[mutation]
            valid_replacement(stage, tmp_path / "mutation", lean_tools, old, new)
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_correspondence_passed"] is (mutation is None)


@pytest.mark.parametrize("change", [False, True])
def test_actual_helper_callee_is_included(tmp_path, lean_tools, change):
    helper = str(source(2, 2, 2))
    prefix, suffix = helper.rsplit("}", 1)
    before = parse_mlir_text(
        prefix.replace("@forward", "@kernel")
        + """func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
          %r = func.call @kernel(%a, %b) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
          return %r : tensor<2x2xf32>
        }}"""
        + suffix
    )
    stage = stage_at(tmp_path, before)
    if change:
        text = (stage / "after.mlir").read_text()
        call_return = [line.strip() for line in text.splitlines() if "func.return" in line][-1]
        target = call_return.split(" : ")[0].removeprefix("func.return ")
        valid_replacement(stage, tmp_path / "mutation", lean_tools, call_return, call_return.replace(target, "%a"))
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_correspondence_passed"] is (not change)
    assert report["evidence"]["reference_functions"] == ["kernel", "forward"]


@pytest.mark.parametrize("mutation", [False, True])
def test_joined_suffix_is_checked_without_global_definedness(tmp_path, lean_tools, mutation):
    first = stage_at(tmp_path, source(2, 2, 2))
    following = tmp_path / "suffix"
    following.mkdir()
    text = (first / "after.mlir").read_text()
    (following / "before.mlir").write_text(text)
    (following / "after.mlir").write_text(text)
    (following / "meta.json").write_text(json.dumps({"boundary": "local-following-boundary"}))
    if mutation:
        returned = next(line.strip() for line in text.splitlines() if "func.return" in line)
        operand = returned.split(" : ")[0].removeprefix("func.return ")
        valid_replacement(following, tmp_path / "mutation", lean_tools, returned, returned.replace(operand, "%a"))
    _, report = invoke([first, following], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_boundary_matched"]
    assert report["evidence"]["complete_correspondence_passed"] is (not mutation)
    assert report["evidence"]["suffix_boundaries"] == [{"index": 0, "matched": not mutation}]


def test_unmodelled_execution_is_never_reported_as_initialized(tmp_path, lean_tools):
    before = source(2, 2, 2)
    text = str(before)
    returned = next(line for line in text.splitlines() if "func.return" in line)
    text = text.replace(returned, '    "frontend.unsupported"() : () -> ()\n' + returned)
    stage = stage_at(tmp_path, parse_mlir_text(text))
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_correspondence_passed"]
    assert report["evidence"]["whole_model_initialized_execution_established"] is False
    assert report["evidence"]["successful_reference_required_for_success_transfer"] is True


def test_actual_receipt_replay_regenerates_complete_checks(tmp_path, lean_tools):
    stage = stage_at(tmp_path, source())
    receipt, report = invoke([stage], tmp_path / "receipt", lean_tools)
    replayed = policy.replay(tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python)
    assert checks.read(Path(replayed["cert"]["path"]))["evidence"] == report["evidence"]
    assert replayed["lean"] == receipt["lean"]
    with pytest.raises(ValueError, match="new destination"):
        policy.replay(tmp_path / "receipt/receipt.json", tmp_path / "receipt", m2m_python=lean_tools.m2m_python)


@pytest.mark.parametrize("forgery", ["premise_type", "whole_model_claim", "source_roster"])
def test_rehashed_report_cannot_change_claim_or_premise_types(tmp_path, lean_tools, forgery):
    stage = stage_at(tmp_path, source())
    receipt, report = invoke([stage], tmp_path / "receipt", lean_tools)
    forged = copy.deepcopy(report)
    if forgery == "premise_type":
        forged["evidence"]["successful_reference_required_for_success_transfer"] = 1
    elif forgery == "whole_model_claim":
        forged["evidence"]["whole_model_initialized_execution_established"] = True
    else:
        forged["evidence"]["source_functions"].append("invented")
    path = Path(receipt["cert"]["path"])
    path.write_text(json.dumps(forged))
    receipt["cert"] = _file_ref(path)
    receipt_path = tmp_path / "receipt/receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="replay differs"):
        policy.replay(receipt_path, tmp_path / "replay", m2m_python=lean_tools.m2m_python)


@pytest.mark.parametrize("integer_left", [False, True])
def test_complete_changed_output_agrees_with_native_and_ordered_recipe(
    tmp_path, lean_tools, fidelity_tools, integer_left
):
    import numpy as np

    case = CASES["interleaved"]
    stage = stage_at(tmp_path, case.source(integer_left=integer_left))
    _, report = invoke([stage], tmp_path / "receipt", lean_tools)
    assert report["evidence"]["complete_correspondence_passed"]
    rng = np.random.default_rng(20261022)
    left = (
        rng.integers(-128, 128, case.shape(case.left), dtype=np.int8)
        if integer_left
        else rng.uniform(-3, 3, case.shape(case.left)).astype(np.float32)
    )
    right = rng.uniform(-5, 5, case.shape(case.right)).astype(np.float32)
    left_bits = [str(int(value) % 256) for value in left.flat] if integer_left else words(left)
    document = Path(report["program_exports"][1]["path"])
    samples = [[left_bits, words(right)]]
    expected = [words(case.recipe(left, right, integer_left=integer_left))]
    work = tmp_path / "native"
    work.mkdir()
    process, lean = evaluate(document, samples, work)
    assert process.returncode == 0, process.stdout + process.stderr
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    diff.assert_equal(lean, expected, outputs, samples)
    native = diff.execute_mlir(
        document.with_suffix(".mlir"), samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert native["results"] == expected
    (work / "ordered-recipe-results.json").write_text(json.dumps(expected) + "\n")
