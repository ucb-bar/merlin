"""Production-pass coverage, full-output checking, and adversarial mutations."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
from lean_overrank_cases import actual, fixture
from lean_overrank_native import CASES

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.export import export_file
from merlin.verify.lean.overrank import prepare, replay, run_checker, source_export, write


@pytest.fixture(scope="module")
def compiler():
    path = Path(os.environ.get("MERLIN_M2M_PYTHON", "/home/jeremy/Projects/model2MLIR/.venv/bin/python"))
    if not path.is_file():
        pytest.fail("native compiler Python is required for Overrank qualification")
    prepare()
    return path


@pytest.mark.parametrize("batch,k,n", CASES)
@pytest.mark.parametrize("kind", ["f32", "f64", "f16", "bf16"])
def test_actual_pass_shapes(tmp_path, compiler, batch, k, n, kind):
    before, after, count = actual(fixture(batch, k, n, kind))
    assert count == 1
    write(tmp_path / "before.json", source_export(before))
    (tmp_path / "after.mlir").write_text(after)
    export_file(tmp_path / "after.mlir", tmp_path / "after.json", m2m_python=compiler)
    report = run_checker(tmp_path / "before.json", tmp_path / "after.json")
    assert report["checked"] and report["covered_sites"] == 1
    assert report["sites"][0]["complete_kernel_execution_checked"]
    assert not report["literal_source_equivalence_claimed"]


def test_nested_regions_and_all_sites(tmp_path, compiler):
    before, after, count = actual(fixture(nested=True))
    assert count == 2
    for label, text in [("before", before), ("after", after)]:
        write(tmp_path / f"{label}.json", source_export(text))
    report = run_checker(tmp_path / "before.json", tmp_path / "after.json")
    assert report["checked"] and report["covered_sites"] == 2


def test_surrounding_function_and_internal_call(tmp_path, compiler):
    source = fixture(batch=(1, 2), k=3, n=2)
    from lean_overrank_cases import render
    from xdsl.dialects import func
    from xdsl.ir import Block, Region

    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    original = next(op for op in module.walk() if op.name == "func.func")
    types = [v.type for v in original.body.block.args]
    block = Block(arg_types=types)
    call = func.CallOp("subject", block.args, [types[-1]])
    block.add_ops([call, func.ReturnOp(call.results[0])])
    module.body.block.add_op(func.FuncOp("entry", (types, [types[-1]]), Region(block)))
    before, after, count = actual(render(module))
    for label, text in [("before", before), ("after", after)]:
        write(tmp_path / f"{label}.json", source_export(text))
    assert count == 1 and run_checker(tmp_path / "before.json", tmp_path / "after.json")["checked"]


MUTATIONS = [
    "lhs_map",
    "rhs_map",
    "out_map",
    "map_symbols",
    "iterator_order",
    "iterator_kind",
    "mul_op",
    "add_order",
    "mul_operand",
    "yield_operand",
    "mul_flag",
    "add_flag",
    "zero_bits",
    "fill_input",
    "fill_yield",
    "fill_segment",
    "empty_shape",
    "result_type",
    "region_argument_type",
    "region_argument_count",
    "generic_input",
    "generic_destination",
    "generic_segment",
    "return_old_destination",
    "extra_operation",
    "missing_site",
    "extra_function",
    "module_attribute",
    "source_k_mismatch",
    "source_rhs_mismatch",
    "source_flag",
    "source_body",
    "source_map",
    "source_initializer_type",
    "target_ssa_duplicate",
    "call_signature",
]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_adversarial_rejection(tmp_path, compiler, mutation):
    before, after, _ = actual(fixture(batch=(1, 2), k=3, n=2))
    source, target = source_export(before), source_export(after)
    block = target["module"]["ops"][0]["regions"][0]["blocks"][0]
    empty, zero, fill, gen, ret = block["ops"]
    body = gen["regions"][0]["blocks"][0]
    mul, add, yieldop = body["ops"]
    maps = gen["attrs"]["indexing_maps"]["elements"]
    if mutation in {"lhs_map", "rhs_map", "out_map"}:
        i = {"lhs_map": 0, "rhs_map": 1, "out_map": 2}[mutation]
        maps[i]["results"][0] = {"kind": "dim", "pos": 2}
    elif mutation == "map_symbols":
        maps[0]["symbols"] = 1
    elif mutation == "iterator_order":
        gen["attrs"]["iterator_types"]["elements"].reverse()
    elif mutation == "iterator_kind":
        gen["attrs"]["iterator_types"]["elements"][-1]["value"] = "parallel"
    elif mutation == "mul_op":
        mul["name"] = "arith.addf"
    elif mutation == "add_order":
        add["operands"].reverse()
    elif mutation == "mul_operand":
        mul["operands"][1] = mul["operands"][0]
    elif mutation == "yield_operand":
        yieldop["operands"] = [mul["results"][0]["id"]]
    elif mutation in {"mul_flag", "add_flag"}:
        (mul if mutation == "mul_flag" else add)["attrs"]["fastmath"]["value"] = "contract"
    elif mutation == "zero_bits":
        zero["attrs"]["value"]["bits"] = "80000000"
    elif mutation == "fill_input":
        fill["operands"][0] = block["args"][0]["id"]
    elif mutation == "fill_yield":
        fill["regions"][0]["blocks"][0]["ops"][0]["operands"] = [fill["regions"][0]["blocks"][0]["args"][1]["id"]]
    elif mutation == "fill_segment":
        fill["attrs"]["operandSegmentSizes"]["values"] = ["0", "2"]
    elif mutation == "empty_shape":
        empty["results"][0]["type"]["shape"][0] = 2
    elif mutation == "result_type":
        gen["results"][0]["type"]["element"]["format"] = "f64"
    elif mutation == "region_argument_type":
        body["args"][0]["type"]["format"] = "f64"
    elif mutation == "region_argument_count":
        body["args"].pop()
    elif mutation == "generic_input":
        gen["operands"][0] = block["args"][2]["id"]
    elif mutation == "generic_destination":
        gen["operands"][2] = empty["results"][0]["id"]
    elif mutation == "generic_segment":
        gen["attrs"]["operandSegmentSizes"]["values"] = ["1", "2"]
    elif mutation == "return_old_destination":
        ret["operands"] = [block["args"][2]["id"]]
    elif mutation == "extra_operation":
        block["ops"].insert(0, copy.deepcopy(zero))
    elif mutation == "missing_site":
        block["ops"] = [ret]
    elif mutation == "extra_function":
        target["module"]["ops"].append(copy.deepcopy(target["module"]["ops"][0]))
    elif mutation == "module_attribute":
        target["module"]["attrs"]["other"] = {"kind": "string", "value": "changed"}
    elif mutation == "source_k_mismatch":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][0]["type"]["shape"][-1] = 4
    elif mutation == "source_rhs_mismatch":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][1]["type"]["shape"][1] = 4
    elif mutation == "source_flag":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]["attrs"][
            "fastmath"
        ]["value"] = "reassoc"
    elif mutation == "source_body":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]["regions"][0]["blocks"][0]["ops"][1][
            "operands"
        ].reverse()
    elif mutation == "source_map":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]["attrs"]["indexing_maps"]["elements"][0][
            "results"
        ][0]["pos"] = 1
    elif mutation == "source_initializer_type":
        source["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][2]["type"]["shape"][0] = 3
    elif mutation == "target_ssa_duplicate":
        block["args"][1]["id"] = block["args"][0]["id"]
    elif mutation == "call_signature":
        target["module"]["ops"][0]["attrs"]["function_type"]["value"]["results"] = []
    write(tmp_path / "before.json", source)
    write(tmp_path / "after.json", target)
    assert not run_checker(tmp_path / "before.json", tmp_path / "after.json")["checked"], mutation


def test_kernel_and_definedness_theorems():
    from merlin.verify.lean.overrank import run

    path = repo_root() / "merlin/tests/fixtures/lean_overrank.lean"
    run(["lake", "env", "lean", str(path)])


@pytest.mark.slow
def test_retained_receipts_replay(tmp_path, compiler):
    base = artifacts_dir() / "verification/merlin-passes/frontend/overrank-matmul"
    receipts = sorted((base / "boundaries").glob("*/receipt.json"))
    assert len(receipts) >= 2, "retained model and positive fixtures must be generated first"
    for i, path in enumerate(receipts):
        result = replay(path, tmp_path / str(i), compiler_python=compiler)
        assert result["fresh"] and result["matches_recorded"] and result["checked"], result


def test_nan_payload_counterexample_preserved():
    import json

    path = (
        artifacts_dir()
        / "verification/merlin-passes/frontend/overrank-matmul/counterexamples/initial-nan-payload/comparison.json"
    )
    report = json.loads(path.read_text())
    assert report["bit_mismatches"]
    assert all(index == 4 for index, _, _ in report["bit_mismatches"])


def test_receipt_rejects_stale_evidence(tmp_path, compiler):
    from merlin.verify.lean.overrank import check_boundary

    before, after, _ = actual(fixture(batch=(1, 1), k=1, n=1))
    (tmp_path / "in.mlir").write_text(before)
    (tmp_path / "out.mlir").write_text(after)
    dest = tmp_path / "evidence"
    receipt = check_boundary(tmp_path / "in.mlir", tmp_path / "out.mlir", dest, compiler_python=compiler)
    assert receipt["checked"] and receipt["verdict"] == "abstracted" and not receipt["intent_approved"]
    (dest / "after.mlir").write_text(after + "\n")
    result = replay(dest / "receipt.json", tmp_path / "replay", compiler_python=compiler)
    assert not result["fresh"] and not result["matches_recorded"]


def test_nested_authoritative_intake_and_replay(tmp_path, compiler):
    from merlin.verify.lean.overrank import check_boundary

    before, after, count = actual(fixture(nested=True))
    assert count == 2
    (tmp_path / "in.mlir").write_text(before)
    (tmp_path / "out.mlir").write_text(after)
    dest = tmp_path / "evidence"
    receipt = check_boundary(tmp_path / "in.mlir", tmp_path / "out.mlir", dest, compiler_python=compiler)
    assert receipt["checked"] and receipt["covered_sites"] == 2
    result = replay(dest / "receipt.json", tmp_path / "replay", compiler_python=compiler)
    assert result["fresh"] and result["matches_recorded"] and result["checked"]
