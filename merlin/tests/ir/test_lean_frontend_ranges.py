"""Bounds from actual SSA producers, compiler-valid refusals, and native fidelity.

The actual attention cut has an initialized mask input contract. It is deliberately
not counted as a continuation past the still-unproved position embedding read.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_lean_frontend_mask_runtime import candidate
from test_lean_frontend_runtime_checkpoint import driver, exported

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff
from merlin.verify.lean import frontend_runtime as runtime
from merlin.xdsl_dialects._common import text as to_text


def independent_gather(n: int, *, row=0, offset=0, scale=1, flags="", uninitialized=False) -> str:
    yielded = "%old" if uninitialized else "%v"
    return f"""builtin.module {{func.func @subject(%mask: tensor<1x{n}xi1>) -> tensor<{n}xi1> {{
      %empty = tensor.empty() : tensor<{n}xi64>
      %cols = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
        outs(%empty : tensor<{n}xi64>) {{ ^bb0(%old: i64):
          %i = linalg.index 0 : index
          %j = arith.index_cast %i : index to i64
          %one = arith.constant {scale} : i64
          %m = arith.muli %j, %one{flags} : i64
          %zero = arith.constant {offset} : i64
          %v = arith.addi %zero, %m : i64
          linalg.yield {yielded} : i64
        }} -> tensor<{n}xi64>
      %row = arith.constant {row} : i64
      %rows = tensor.splat %row : tensor<{n}xi64>
      %outempty = tensor.empty() : tensor<{n}xi1>
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
        ins(%rows, %cols : tensor<{n}xi64>, tensor<{n}xi64>) outs(%outempty : tensor<{n}xi1>) {{
        ^bb0(%r: i64, %c: i64, %old: i1):
          %ri = arith.index_cast %r : i64 to index
          %ci = arith.index_cast %c : i64 to index
          %bit = tensor.extract %mask[%ri, %ci] : tensor<1x{n}xi1>
          linalg.yield %bit : i1
        }} -> tensor<{n}xi1>
      return %out : tensor<{n}xi1> }} }}"""


def actual_attention() -> str:
    """Clone all 35 actual operations, retaining every nested operation and use."""
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    source = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(source.read_text())
    forward = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
    selected = list(forward.body.block.ops)[122:157]
    nested = [child for op in selected for child in op.walk()]
    local = {v for op in nested for v in op.results}
    local.update(a for op in nested for r in op.regions for b in r.blocks for a in b.args)
    free = list(dict.fromkeys(v for op in nested for v in op.operands if v not in local))
    block = Block(arg_types=[v.type for v in free])
    mapping = dict(zip(free, block.args, strict=True))
    for operation in selected:
        block.add_op(operation.clone(value_mapper=mapping))
    result = list(block.ops)[-1].results[0]
    block.add_op(func.ReturnOp(result))
    subject = func.FuncOp.from_region("subject", [v.type for v in free], [result.type], Region(block))
    isolated = ModuleOp([subject])
    isolated.verify()
    return to_text(isolated, generic=True)


def checked(source: str, work: Path, tools, *, mode="bounds_defined", **extra) -> dict:
    path = exported(source, work, tools)
    return driver({"mode": mode, "function": "subject", "after_export": str(path), **extra}, work, tools)


def test_actual_attention_bounds_and_source_projection(tmp_path, fidelity_tools):
    source = actual_attention()
    assert checked(source, tmp_path / "cut", fidelity_tools)["accepted"]
    assert checked(source, tmp_path / "projection", fidelity_tools, mode="bounds_slice", start=0, count=35)["accepted"]
    # Old initialization-only rules cannot establish the actual index bounds.
    assert not checked(source, tmp_path / "old", fidelity_tools, mode="defined")["accepted"]
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    path = exported(capture.read_text(), tmp_path / "full-source", fidelity_tools)
    answer = driver(
        {"mode": "bounds_slice", "function": "forward", "after_export": str(path), "start": 122, "count": 35},
        tmp_path / "actual-source",
        fidelity_tools,
    )
    assert answer["accepted"] and not answer["whole_function_defined"]
    assert answer["input_domain"] == "typed initialized cut inputs"


@pytest.mark.parametrize("n", [0, 1, 8])
def test_independent_bounds_and_native(tmp_path, fidelity_tools, n):
    source = independent_gather(n)
    assert checked(source, tmp_path / "proof", fidelity_tools)["accepted"]
    samples = [[[str(i % 2) for i in range(n)]], [["1"] * n], [["0"] * n]]
    lean, document, path = candidate(source, samples, tmp_path / "execute", fidelity_tools)
    inputs, outputs = diff.signature(document, "subject")
    native = diff.execute_mlir(path, samples, tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs)
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert [x["values"][0]["bits"] for x in lean] == [x[0] for x in samples]


def test_actual_attention_native(tmp_path, fidelity_tools):
    source = actual_attention()
    samples = [[[str(i % 2) for i in range(1024)]], [["1"] * 1024], [["0"] * 1024]]
    lean, document, path = candidate(source, samples, tmp_path / "execute", fidelity_tools)
    inputs, outputs = diff.signature(document, "subject")
    native = diff.execute_mlir(path, samples, tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs)
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert [x["values"][0]["bits"] for x in lean] == [x[0] for x in samples]


@pytest.mark.parametrize(
    "change",
    [
        {"row": 1},
        {"row": -1},
        {"offset": 1},
        {"offset": -1},
        {"scale": 2},
        {"flags": " overflow<nsw>"},
        {"uninitialized": True},
    ],
)
def test_compiler_valid_bounds_mutations_rejected_by_lean(tmp_path, fidelity_tools, change):
    # exported() runs the authoritative compiler verifier before the actual Lean checker.
    assert not checked(independent_gather(8, **change), tmp_path, fidelity_tools)["accepted"]


def test_unknown_tensor_indices_are_not_an_input_bound(tmp_path, fidelity_tools):
    source = """builtin.module {func.func @subject(%mask: tensor<8xi1>, %indices: tensor<8xi64>) -> tensor<8xi1> {
      %empty = tensor.empty() : tensor<8xi1>
      %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
        ins(%indices : tensor<8xi64>) outs(%empty : tensor<8xi1>) { ^bb0(%i: i64, %old: i1):
          %idx = arith.index_cast %i : i64 to index
          %v = tensor.extract %mask[%idx] : tensor<8xi1>
          linalg.yield %v : i1 } -> tensor<8xi1>
      return %r : tensor<8xi1> }}"""
    assert not checked(source, tmp_path, fidelity_tools)["accepted"]


def test_actual_attention_out_of_bounds_mutation(tmp_path, fidelity_tools):
    from xdsl.dialects import arith, builtin

    module = parse_mlir_text(actual_attention())
    generators = [
        op
        for op in module.walk()
        if op.name == "linalg.generic" and any(child.name == "linalg.index" for child in op.walk())
    ]
    # The second generator produces the one-element row-index tensor. Change its
    # zero offset to one, keeping every type, SSA edge and MLIR verifier constraint valid.
    constants = [op for op in generators[1].walk() if isinstance(op, arith.ConstantOp)]
    constants[-1].properties["value"] = builtin.IntegerAttr(1, builtin.i64)
    assert not checked(to_text(module, generic=True), tmp_path, fidelity_tools)["accepted"]


def test_actual_boundary_receipt_binds_checked_slices(tmp_path, fidelity_tools):
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(to_text(module, generic=True))
    assert fix_bool_fptosi(module) == 1
    after.write_text(to_text(module, generic=True))
    receipt = runtime.verify_candidate(
        before,
        after,
        tmp_path / "receipt",
        function="forward",
        start=91,
        compiler_python=fidelity_tools.compiler_python,
        lake=fidelity_tools.lake,
        bounds_slices=((122, 35),),
        suffix=(after,),
    )
    assert receipt["status"] == "verified_candidate", receipt.get("reason")
    assert receipt["answer"]["bounds_slices"] == [
        {
            "start": 122,
            "count": 35,
            "accepted": True,
            "input_domain": "typed initialized cut inputs",
            "theorem": "MerlinVerify.Rules.FrontendRangeProjection.check_sound",
        }
    ]
    runtime.validate_receipt(receipt)
    wrong = copy.deepcopy(receipt)
    wrong["bounds_slices"] = []
    with pytest.raises(ValueError, match="receipt order or checkpoint"):
        runtime.validate_receipt(wrong)

    # A cut requiring unproved position-index bounds cannot be silently added to
    # an accepted repair/prefix receipt. Lean refuses it, preserving the claim.
    work = tmp_path / "position-refusal"
    work.mkdir()
    request = work / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": runtime.REQUEST,
                "index_bits": 64,
                "limit": 1024,
                "mode": "boundary",
                "function": "forward",
                "start": 91,
                "before_export": receipt["artifacts"][0]["export"]["path"],
                "after_export": receipt["artifacts"][1]["export"]["path"],
                "suffix_exports": [],
                "bounds_slices": [{"start": 116, "count": 2}],
            }
        )
    )
    executable = merlin_dir().parent / "out/build/lean/build/bin/merlin-lean-frontend-runtime"
    answer = json.loads(runtime._run([str(executable), str(request)], work, "checker").stdout)
    assert answer["repair"] and answer["checkpoint"]
    assert not answer["accepted"] and not answer["bounds_slices"][0]["accepted"]


def independent_positions() -> str:
    """Independent integer bucket-grid analogue; no assumption on actual SmolVLA buckets."""
    return """builtin.module {func.func @subject(%mask: tensor<16xi1>) -> tensor<16xi1> {
      %empty = tensor.empty() : tensor<4xi64>
      %coords = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
        outs(%empty : tensor<4xi64>) { ^bb0(%old: i64):
          %i = linalg.index 0 : index
          %v = arith.index_cast %i : index to i64
          linalg.yield %v : i64 } -> tensor<4xi64>
      %four = arith.constant 4 : i64
      %scaledempty = tensor.empty() : tensor<4xi64>
      %scaled = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]} ins(%coords : tensor<4xi64>) outs(%scaledempty : tensor<4xi64>) {
          ^bb0(%v: i64, %old: i64):
          %m = arith.muli %v, %four : i64 linalg.yield %m : i64 } -> tensor<4xi64>
      %rows = tensor.expand_shape %scaled [[0,1]] output_shape [4,1] : tensor<4xi64> into tensor<4x1xi64>
      %cols = tensor.expand_shape %coords [[0,1]] output_shape [1,4] : tensor<4xi64> into tensor<1x4xi64>
      %gridempty = tensor.empty() : tensor<4x4xi64>
      %grid = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,0)>, affine_map<(d0,d1)->(0,d1)>,
        affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%rows, %cols : tensor<4x1xi64>, tensor<1x4xi64>) outs(%gridempty : tensor<4x4xi64>) {
          ^bb0(%r: i64, %c: i64, %old: i64):
          %i = arith.addi %r, %c : i64 linalg.yield %i : i64 } -> tensor<4x4xi64>
      %indices = tensor.collapse_shape %grid [[0,1]] : tensor<4x4xi64> into tensor<16xi64>
      %outempty = tensor.empty() : tensor<16xi1>
      %out = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]} ins(%indices : tensor<16xi64>) outs(%outempty : tensor<16xi1>) {
          ^bb0(%i: i64, %old: i1):
          %idx = arith.index_cast %i : i64 to index
          %bit = tensor.extract %mask[%idx] : tensor<16xi1>
          linalg.yield %bit : i1 } -> tensor<16xi1>
      return %out : tensor<16xi1> }}"""


def test_independent_position_arithmetic_and_native(tmp_path, fidelity_tools):
    source = independent_positions()
    assert checked(source, tmp_path / "proof", fidelity_tools)["accepted"]
    samples = [[[str(i % 2) for i in range(16)]], [["1"] * 16], [["0"] * 16]]
    lean, document, path = candidate(source, samples, tmp_path / "execute", fidelity_tools)
    inputs, outputs = diff.signature(document, "subject")
    native = diff.execute_mlir(path, samples, tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs)
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert [x["values"][0]["bits"] for x in lean] == [x[0] for x in samples]
