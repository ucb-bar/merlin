"""Actual-program initialization, repair composition, and inactive receipt integrity."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean import frontend_runtime as runtime
from merlin.verify.lean.export import ExportError, export_file
from merlin.xdsl_dialects._common import text as to_text


def driver(request: dict, work: Path, tools) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    path = work / "request.json"
    path.write_text(json.dumps({"schema": runtime.REQUEST, "index_bits": 64, "limit": 1024, **request}))
    proc = subprocess.run(
        [str(tools.lake), "env", "lean", "--run", str(merlin_dir() / "lean/FrontendRuntimeCandidate.lean"), str(path)],
        cwd=merlin_dir() / "lean",
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def exported(source: str, work: Path, tools) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    mlir, path = work / "subject.mlir", work / "subject.json"
    mlir.write_text(source)
    # The authoritative compiler parses AND verifies every positive and mutated program first.
    export_file(mlir, path, backend="mlir", m2m_python=tools.compiler_python)
    return path


def checkpoint_sources() -> tuple[str, str]:
    """Actual initial 116 operations, with their original block args, regions and SSA uses."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    module = parse_mlir_text(
        (merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir").read_text()
    )
    result = []
    for repair in (False, True):
        if repair:
            assert fix_bool_fptosi(module) == 1
        forward = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
        original = list(forward.body.block.ops)
        block = Block(arg_types=forward.function_type.inputs)
        mapping = dict(zip(forward.body.block.args, block.args, strict=True))
        for op in original[:116]:
            block.add_op(op.clone(value_mapper=mapping))
        cloned = list(block.ops)
        out, count = cloned[-1].results[0], cloned[-2].results[1]
        block.add_op(func.ReturnOp(out, count))
        projected = ModuleOp(
            [
                func.FuncOp.from_region(
                    "subject", list(forward.function_type.inputs), [out.type, count.type], Region(block)
                )
            ]
        )
        projected.verify()
        result.append(to_text(projected, generic=True))
    return tuple(result)


def test_actual_checkpoint_receipt_and_integrity(tmp_path, fidelity_tools):
    before, after = checkpoint_sources()
    paths = [tmp_path / name for name in ("before.mlir", "after.mlir")]
    for path, text in zip(paths, (before, after), strict=True):
        path.write_text(text)
    receipt = runtime.verify_candidate(
        *paths,
        tmp_path / "evidence",
        function="subject",
        start=91,
        compiler_python=Path(fidelity_tools.compiler_python),
        suffix=(paths[1],),
        lake=fidelity_tools.lake,
    )
    assert receipt["status"] == "verified_candidate", receipt.get("reason")
    assert receipt["answer"]["repair"] and receipt["answer"]["checkpoint"] and receipt["answer"]["suffix"]
    assert not receipt["whole_function_defined"] and not receipt["ordinary_source_equivalence"]
    runtime.validate_receipt(receipt)
    mutations = []
    wrong = copy.deepcopy(receipt)
    wrong["schema"] = "merlin.lean.receipt.v1"
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["configuration"]["limit"] = 1023
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["interpretation"]["sources"]["merlin/lean/MerlinVerify/Rules/FrontendMaskRuntime.lean"] = "0" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["interpretation"]["sources"]["merlin/lean/MerlinVerify/Spec/FrontendIntent.lean"] = "0" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["artifacts"][0], wrong["artifacts"][1] = wrong["artifacts"][1], wrong["artifacts"][0]
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["answer"]["checkpoint"] = False
    mutations.append(wrong)
    wrong = copy.deepcopy(receipt)
    wrong["whole_function_defined"] = True
    mutations.append(wrong)
    for wrong in mutations:
        with pytest.raises(ValueError):
            runtime.validate_receipt(wrong)
    # Recomputing hashes of a forged execution result cannot create proof evidence.
    wrong = copy.deepcopy(receipt)
    wrong["answer"]["inventory"]["operations"] += 1
    forged = tmp_path / "forged-result.json"
    forged.write_text(json.dumps(wrong["answer"]))
    wrong["result"] = runtime._ref(forged)
    with pytest.raises(ValueError, match="actual Lean checker replay differs"):
        runtime.validate_receipt(wrong)
    artifact = Path(receipt["artifacts"][1]["export"]["path"])
    artifact.write_text(artifact.read_text() + "\n")
    with pytest.raises(ValueError, match="stale export"):
        runtime.validate_receipt(receipt)


@pytest.mark.parametrize(
    "body,accepted",
    [
        ("%i = linalg.index 0 : index\n%v = arith.index_cast %i : index to i64\nlinalg.yield %v : i64", True),
        ("linalg.yield %old : i64", False),
    ],
)
def test_independent_index_initialization(tmp_path, fidelity_tools, body, accepted):
    source = """builtin.module {func.func @subject() -> tensor<4xi64> {
      %empty = tensor.empty() : tensor<4xi64>
      %out = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
        outs(%empty : tensor<4xi64>) { ^bb0(%old: i64): BODY } -> tensor<4xi64>
      return %out : tensor<4xi64> }}""".replace("BODY", body)
    path = exported(source, tmp_path, fidelity_tools)
    result = driver({"mode": "defined", "function": "subject", "after_export": str(path)}, tmp_path, fidelity_tools)
    assert result["accepted"] is accepted


@pytest.mark.parametrize("slice_type,offset,accepted", [("tensor<2xi64>", 1, True), ("tensor<2xi64>", 2, True)])
def test_static_slice_initialization_and_bounds(tmp_path, fidelity_tools, slice_type, offset, accepted):
    source = f"""builtin.module {{func.func @subject(%x: tensor<4xi64>) -> {slice_type} {{
      %r = tensor.extract_slice %x[{offset}] [2] [1] : tensor<4xi64> to {slice_type}
      return %r : {slice_type} }} }}"""
    path = exported(source, tmp_path, fidelity_tools)
    result = driver({"mode": "defined", "function": "subject", "after_export": str(path)}, tmp_path, fidelity_tools)
    assert result["accepted"] is accepted


def test_dynamic_slice_is_explicitly_refused(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%x: tensor<4xi64>, %off: index) -> tensor<2xi64> {
      %r = tensor.extract_slice %x[%off] [2] [1] : tensor<4xi64> to tensor<2xi64>
      return %r : tensor<2xi64> }}"""
    path = exported(source, tmp_path, fidelity_tools)
    assert not driver({"mode": "defined", "function": "subject", "after_export": str(path)}, tmp_path, fidelity_tools)[
        "accepted"
    ]


def test_static_out_of_bounds_slice_is_rejected_by_compiler(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%x: tensor<4xi64>) -> tensor<2xi64> {
      %r = tensor.extract_slice %x[3] [2] [1] : tensor<4xi64> to tensor<2xi64>
      return %r : tensor<2xi64> }}"""
    # This is compiler qualification, never counted as a Lean mutation rejection.
    with pytest.raises(ExportError, match="out-of-bounds"):
        exported(source, tmp_path, fidelity_tools)


def test_rank_reducing_slice_initialization(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%x: tensor<1x4xi64>) -> tensor<2xi64> {
      %r = tensor.extract_slice %x[0,1] [1,2] [1,1] : tensor<1x4xi64> to tensor<2xi64>
      return %r : tensor<2xi64> }}"""
    path = exported(source, tmp_path, fidelity_tools)
    assert driver({"mode": "defined", "function": "subject", "after_export": str(path)}, tmp_path, fidelity_tools)[
        "accepted"
    ]


@pytest.mark.parametrize("flags,accepted", [("", True), (" fastmath<nnan>", False)])
def test_comparison_initialization_metadata(tmp_path, fidelity_tools, flags, accepted):
    source = f"""builtin.module {{func.func @subject(%x: f32, %y: f32) -> i1 {{
      %r = arith.cmpf une, %x, %y{flags} : f32
      return %r : i1 }} }}"""
    path = exported(source, tmp_path, fidelity_tools)
    assert (
        driver({"mode": "defined", "function": "subject", "after_export": str(path)}, tmp_path, fidelity_tools)[
            "accepted"
        ]
        is accepted
    )


def test_actual_boundary_mutations_rejected_by_lean(tmp_path, fidelity_tools):
    from xdsl.dialects import arith, builtin

    before, after = checkpoint_sources()
    src = exported(before, tmp_path / "source", fidelity_tools)
    original = parse_mlir_text(after)
    predicate = next(op for op in original.walk() if isinstance(op, arith.CmpfOp))
    cases = {}
    wrong = original.clone()
    cmp = next(op for op in wrong.walk() if isinstance(op, arith.CmpfOp))
    cmp.properties["predicate"] = builtin.IntegerAttr(1, builtin.i64)
    cases["wrong-predicate"] = wrong
    # Positive zero and the exact absence of fast-math are part of acceptance.
    wrong = parse_mlir_text(after.replace("#arith.fastmath<none>", "#arith.fastmath<nnan>", 1))
    cases["fast-math"] = wrong
    wrong = original.clone()
    cmp = next(op for op in wrong.walk() if isinstance(op, arith.CmpfOp))
    cmp.operands = [cmp.operands[1], cmp.operands[1]]
    cases["wrong-source"] = wrong
    wrong = original.clone()
    generic = next(
        op for op in wrong.walk() if op.name == "linalg.generic" and any(o.name == "linalg.index" for o in op.walk())
    )
    returned = generic.regions[0].block.last_op
    # Returning the uninitialized output operand bypasses initialization, yet stays well typed.
    returned.operands = [generic.regions[0].block.args[-1]]
    cases["uninitialized-prefix-output"] = wrong
    assert predicate is not None
    for name, module in cases.items():
        path = exported(to_text(module, generic=True), tmp_path / name, fidelity_tools)
        answer = driver(
            {
                "mode": "boundary",
                "function": "subject",
                "start": 91,
                "before_export": str(src),
                "after_export": str(path),
                "suffix_exports": [],
            },
            tmp_path / name,
            fidelity_tools,
        )
        assert not answer["accepted"], name
        if name == "uninitialized-prefix-output":
            assert not answer["checkpoint"]

    target = exported(after, tmp_path / "target", fidelity_tools)
    suffix = tmp_path / "wrong-predicate/subject.json"
    answer = driver(
        {
            "mode": "boundary",
            "function": "subject",
            "start": 91,
            "before_export": str(src),
            "after_export": str(target),
            "suffix_exports": [str(suffix)],
        },
        tmp_path / "bad-suffix",
        fidelity_tools,
    )
    assert answer["repair"] and answer["checkpoint"]
    assert not answer["suffix"] and not answer["accepted"]


def test_missing_runtime_tools_is_unmeasured(tmp_path):
    receipt = runtime.verify_candidate(
        tmp_path / "absent.mlir",
        tmp_path / "absent2.mlir",
        tmp_path / "evidence",
        function="subject",
        start=0,
        compiler_python=Path("/missing/compiler"),
        lake=Path("/missing/lake"),
    )
    assert receipt["status"] == "unmeasured"
    with pytest.raises(ValueError, match="not verified"):
        runtime.validate_receipt(receipt)
