"""Actual MACC output keeps shared SSA identities and complete call/dataflow joins."""

import json
from pathlib import Path

import pytest
from lean_bf16_f32acc_cases import line_with, operands, replace_once, run_pass
from lean_macc_ssa_cases import called, dependent_sites, initialized_operand, shared_operand

from merlin.verify import compiler_macc
from merlin.verify.lean.bf16_f32acc import run_checker
from merlin.verify.lean.export import export_file


def checked(tools, work: Path, source: str, output: str):
    paths = []
    for name, text in (("before", source), ("after", output)):
        original = work / f"{name}.mlir"
        exported = work / f"{name}.json"
        original.write_text(text)
        export_file(original, exported, backend="mlir", m2m_python=tools.m2m_python)
        paths.append(exported)
    result = run_checker(paths, index_bits=64, timeout_seconds=60)
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
@pytest.mark.parametrize("kind", ["named-shared", "generic-shared", "callee", "dependent"])
def test_actual_shared_and_call_joins_qualify(lean_tools, tmp_path, fmt, kind):
    source = (
        called(fmt)
        if kind == "callee"
        else dependent_sites(fmt)
        if kind == "dependent"
        else shared_operand(fmt, generic=kind == "generic-shared")
    )
    output, count = run_pass(source)
    assert count == (2 if kind == "dependent" else 1)
    result = checked(lean_tools, tmp_path, source, output)
    local = result["pass_boundary"]
    assert local["semantic_match"]
    assert local["site_qualification_passed"]
    assert local["positive_boundary_checked"]
    assert local["covered_sites"] == local["expected_sites"] == count
    assert result["policy_reviewed"] is False and result["verdict"] == "abstracted"


@pytest.mark.parametrize("mutation", ["call-operands", "dependent-input", "shared-input"])
def test_wrong_ssa_or_call_join_refuses(lean_tools, tmp_path, mutation):
    source = (
        called()
        if mutation == "call-operands"
        else dependent_sites()
        if mutation == "dependent-input"
        else shared_operand()
    )
    output, _ = run_pass(source)
    if mutation == "call-operands":
        call = line_with(output, '"func.call"')
        a, b = operands(call)
        output = replace_once(output, call, call.replace(f"({a}, {b})", f"({b}, {a})"))
    else:
        reductions = [
            line for line in output.splitlines() if '"linalg.generic"' in line and "iterator_type<reduction>" in line
        ]
        use = reductions[-1]
        a, b, initial = operands(use)
        # Both source operands intentionally name the same tensor. Replacing one
        # with another same-typed tensor must never become a packed shared input.
        other = operands(reductions[0])[0] if mutation == "dependent-input" else "%b"
        output = replace_once(output, use, use.replace(f"({a}, {b}, {initial})", f"({a}, {other}, {initial})"))
    result = checked(lean_tools, tmp_path, source, output)
    assert not result["pass_boundary"]["positive_boundary_checked"]
    assert not result["pass_boundary"]["semantic_match"]


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_initialized_operand_keeps_its_actual_producer(lean_tools, tmp_path, fmt, generic, side):
    source = initialized_operand(fmt, generic=generic, side=side)
    output, count = run_pass(source)
    assert count == 1
    result = checked(lean_tools, tmp_path, source, output)
    assert result["pass_boundary"]["semantic_match"]
    assert result["pass_boundary"]["site_qualification_passed"]
    assert result["pass_boundary"]["positive_boundary_checked"]
    assert result["policy_reviewed"] is False


def test_initialized_source_cannot_be_replaced_by_an_arbitrary_tensor(lean_tools, tmp_path):
    source = initialized_operand()
    output, _ = run_pass(source)
    reduction = line_with(output, '"linalg.generic"', "iterator_type<reduction>")
    left, right, initial = operands(reduction)
    output = replace_once(
        output, reduction, reduction.replace(f"({left}, {right}, {initial})", f"(%a, {right}, {initial})")
    )
    result = checked(lean_tools, tmp_path, source, output)
    assert not result["pass_boundary"]["semantic_match"]
    assert not result["pass_boundary"]["positive_boundary_checked"]


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
@pytest.mark.parametrize("initialized", [False, True])
def test_shared_identity_reaches_live_delivery_and_fresh_replay(pass_gate_tools, tmp_path, fmt, initialized):
    source = initialized_operand(fmt, side="both") if initialized else shared_operand(fmt)
    output, _ = run_pass(source)
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(source)
    after.write_text(output)
    completion = compiler_macc.check_boundary(
        before,
        after,
        tmp_path / "check",
        sdk_bin=pass_gate_tools.sdk_bin,
        compiler_python=pass_gate_tools.compiler_python,
    )
    assert compiler_macc.deliver(completion) == output
    detail = compiler_macc.validate_completion(completion)
    assert detail["policy_reviewed"] is False
    assert all(
        "MerlinVerify.Check.Bf16F32AccOperands." + name in detail["audit"]["axioms"]
        for name in ("pack_nodup", "same_sharing", "bind_sound")
    )
    replayed = compiler_macc.replay(
        completion.report,
        tmp_path / "replay",
        sdk_bin=pass_gate_tools.sdk_bin,
        compiler_python=pass_gate_tools.compiler_python,
    )
    assert compiler_macc.deliver(replayed) == output
