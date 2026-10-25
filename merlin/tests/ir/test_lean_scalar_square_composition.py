"""Typed composition claims, actual nested calls and fail-closed approval/definedness status."""

import json
from pathlib import Path

import pytest
from lean_scalar_square_cases import actual

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.export import export_file
from merlin.verify.lean.scalar_square import _run
from merlin.verify.lean.scalar_square_composition import prepare, run


def program(kind="f32"):
    return f"""module {{
      func.func private @leaf(%x: {kind}) -> {kind} {{
        %two = arith.constant 2.0 : {kind}
        %squared = math.powf %x, %two : {kind}
        return %squared : {kind}
      }}
      func.func private @middle(%x: {kind}) -> {kind} {{
        %r = func.call @leaf(%x) : ({kind}) -> {kind}
        return %r : {kind}
      }}
      func.func @subject(%input: tensor<2x{kind}>) -> tensor<2x{kind}> {{
        %two = arith.constant 2.0 : {kind}
        %empty = tensor.empty() : tensor<2x{kind}>
        %result = linalg.generic {{
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        }} ins(%input : tensor<2x{kind}>) outs(%empty : tensor<2x{kind}>) {{
          ^bb0(%x: {kind}, %old: {kind}):
            %squared = math.powf %x, %two : {kind}
            %r = func.call @middle(%squared) : ({kind}) -> {kind}
            linalg.yield %r : {kind}
        }} -> tensor<2x{kind}>
        return %result : tensor<2x{kind}>
      }}
    }}"""


@pytest.fixture(scope="module", autouse=True)
def composition_driver():
    prepare()


def report(tmp_path, lean_tools, before, after):
    paths = []
    for label, text in (("before", before), ("after", after)):
        source = tmp_path / f"{label}.mlir"
        source.write_text(text)
        exported = tmp_path / f"{label}.json"
        export_file(source, exported, backend="mlir", m2m_python=lean_tools.m2m_python)
        paths.append(exported)
    return run(*paths)


@pytest.mark.parametrize("kind", ["f32", "f64"])
def test_nested_region_and_two_calls(tmp_path, lean_tools, kind):
    before, after, count = actual(program(kind))
    assert count == 2
    result = report(tmp_path, lean_tools, before, after)
    assert result["program_structure_checked"]
    assert result["function_inventory_checked"]
    assert result["covered_sites"] == 2 and result["call_fuel"] == 4
    assert result["unsupported_runtime_types"] == 0
    assert not result["strict_source_claim"]["instantiated"]
    assert result["strict_source_claim"]["source_typing"] == "not_discharged"
    assert result["strict_source_claim"]["source_definedness"] == "not_discharged"
    assert not result["multiplication_policy_claim"]["approved"]
    assert not result["multiplication_policy_claim"]["requires_pow_square_law"]
    assert not result["multiplication_policy_claim"]["original_source_equivalence_claimed"]


@pytest.mark.parametrize("mutation", ["callee", "operand", "visibility", "extra_function", "missing_function"])
def test_call_and_inventory_corruptions(tmp_path, lean_tools, mutation):
    from lean_scalar_square_cases import render
    from xdsl.dialects import func

    from merlin.frontends.linalg_mlir import parse_mlir_text

    before, after, _ = actual(program())
    module = parse_mlir_text(after)
    functions = [op for op in module.walk() if isinstance(op, func.FuncOp)]
    if mutation == "callee":
        call = next(
            op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "middle"
        )
        from xdsl.dialects.builtin import SymbolRefAttr

        call.properties["callee"] = SymbolRefAttr("leaf")
    elif mutation == "operand":
        call = next(
            op for op in module.walk() if isinstance(op, func.CallOp) and op.callee.root_reference.data == "middle"
        )
        call.operands = (call.parent.args[1],)
    elif mutation == "visibility":
        del functions[0].properties["sym_visibility"]
    elif mutation == "extra_function":
        from xdsl.dialects.builtin import f32
        from xdsl.ir import Block, Region

        block = Block(arg_types=[f32])
        block.add_op(func.ReturnOp(block.args[0]))
        module.body.block.add_op(func.FuncOp("unused", ([f32], [f32]), Region(block)))
    else:
        from xdsl.dialects.builtin import SymbolRefAttr

        for op in module.walk():
            if isinstance(op, func.CallOp) and op.callee.root_reference.data == "middle":
                op.properties["callee"] = SymbolRefAttr("leaf")
        module.body.block.erase_op(functions[1])
    module.verify()
    result = report(tmp_path, lean_tools, before, render(module))
    assert not result["program_structure_checked"]
    assert not result["boundary_checked"]


def test_lean_type_shadow_approval_and_definedness_obligations():
    fixture = repo_root() / "merlin/tests/fixtures/lean_scalar_square_composition.lean"
    output = _run(["lake", "env", "lean", str(fixture)])
    assert "sorryAx" not in output


@pytest.mark.slow
@pytest.mark.parametrize("model", ["SY_model_smolvla", "SY_model_tiny_llama"])
def test_retained_program_claims_replay(tmp_path, lean_tools, model):
    from merlin.verify.lean.scalar_square_composition import replay_program

    path = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square/composition" / model / "receipt.json"
    result = replay_program(path, tmp_path / "replay", compiler_python=lean_tools.m2m_python)
    assert result["fresh"] and result["matches_recorded"]
    assert not result["strict_source_claim_instantiated"]
    assert not result["policy_claim_instantiated"]


def test_forged_approval_and_definedness_report_is_recomputed(tmp_path, lean_tools):
    from merlin.verify.lean.check import _file_ref
    from merlin.verify.lean.scalar_square import check_stage
    from merlin.verify.lean.scalar_square_composition import check, replay_program

    before, after, _ = actual(program())
    stage = tmp_path / "stage"
    stage.mkdir()
    for name, text in (("before", before), ("after", after)):
        (stage / f"{name}.mlir").write_text(text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "prepare-scalar-squares"}))
    check_stage(stage, tmp_path / "base", compiler_python=lean_tools.m2m_python)
    directory = tmp_path / "composition"
    receipt = check(tmp_path / "base/receipt.json", directory, compiler_python=lean_tools.m2m_python)
    path = Path(receipt["report"]["path"])
    forged = json.loads(path.read_text())
    forged["multiplication_policy_claim"].update(approved=True, instantiated=True, policy_definedness="discharged")
    forged["strict_source_claim"].update(instantiated=True, numerical_law="discharged", source_typing="discharged")
    path.write_text(json.dumps(forged))
    receipt.update(report=_file_ref(path), policy_approved=True, policy_claim_instantiated=True)
    (directory / "receipt.json").write_text(json.dumps(receipt))
    result = replay_program(
        directory / "receipt.json", tmp_path / "forged-replay", compiler_python=lean_tools.m2m_python
    )
    assert result["fresh"] and not result["matches_recorded"] and not result["claim_metadata_matches"]
    assert not result["policy_claim_instantiated"]
