"""Closed fills needed by tensor transport stay compute-only tensor kernels."""

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain
from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

SOURCE = r"""module {
func.func @forward(%x:tensor<3xi32>)->(tensor<i32>,tensor<i32>) {
 %e=tensor.empty():tensor<i32>
 %z=arith.constant 0:i32
 %init=linalg.fill ins(%z:i32) outs(%e:tensor<i32>)->tensor<i32>
 %sum=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}
 ins(%x:tensor<3xi32>) outs(%init:tensor<i32>) {
 ^bb0(%value:i32,%acc:i32):
  %v=arith.addi %value,%acc:i32
  linalg.yield %v:i32
 }->tensor<i32>
 func.return %init,%sum:tensor<i32>,tensor<i32>
}}
"""


def test_default_outlining_and_seed_producer_cloning_are_unchanged():
    baseline = outline_dispatches(parse_mlir_text(SOURCE))
    explicit = outline_dispatches(parse_mlir_text(SOURCE), outline_literal_fills=False)
    assert str(baseline.module) == str(explicit.module)
    assert len(baseline.dispatches) == 1


def test_retained_literal_is_closed_but_normal_root_still_clones_its_seed():
    result = outline_dispatches(parse_mlir_text(SOURCE), outline_literal_fills=True)
    assert [d.root_op for d in result.dispatches] == ["linalg.fill", "linalg.generic"]
    kernels = [op for op in result.module.body.block.ops if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    assert not kernels[0].function_type.inputs.data
    assert len(kernels[1].function_type.inputs.data) == 1  # x; never the writable seed tensor
    assert [str(t) for t in kernels[0].function_type.outputs] == ["tensor<i32>"]
    assert any(op.name == "linalg.fill" for op in kernels[1].walk())
    result.module.verify()


def test_selected_deployable_path_keeps_both_literal_and_reduction_on_workers():
    planned = plan_deployable_chain(
        parse_mlir_text(SOURCE),
        worker_slots=(0, 1),
        worker_count=2,
        workspace_budget_bytes=1024,
        alignment_bytes=64,
        data_preparation=True,
    )
    assert len(planned.dispatches) == 2
    assert planned.data_preparation["literal_fill_kernels"] == [planned.dispatches[0]["symbol"]]
    assert planned.to_dict()["controller_arithmetic"] is False
    assert [len(step["inputs"]) for step in planned.dispatches] == [0, 1]


@pytest.mark.parametrize("change", ["body", "unknown", "quant_inner", "borrowed_destination"])
def test_unknown_or_borrowed_literal_fill_is_not_promoted(change):
    source = SOURCE
    if change == "body":
        # Keep a valid body type while changing the actual fill scalar law.
        m = parse_mlir_text(source)
        fill = next(op for op in m.walk() if op.name == "linalg.fill")
        fill.body.block.last_op.operands = (fill.body.block.args[1],)
    elif change == "borrowed_destination":
        source = source.replace("%x:tensor<3xi32>", "%x:tensor<3xi32>,%dest:tensor<i32>", 1)
        source = source.replace("outs(%e:tensor<i32>)", "outs(%dest:tensor<i32>)", 1)
        m = parse_mlir_text(source)
    else:
        m = parse_mlir_text(source)
        next(op for op in m.walk() if op.name == "linalg.fill").attributes[
            "test.unknown" if change == "unknown" else "prov.quant_inner"
        ] = StringAttr("unproved")
    result = outline_dispatches(m, outline_literal_fills=True)
    assert [d.root_op for d in result.dispatches] == ["linalg.generic"]


@pytest.mark.parametrize("value", [1, None, "true"])
def test_literal_fill_selection_is_explicit_boolean(value):
    with pytest.raises(OutlineError, match="explicit boolean"):
        outline_dispatches(parse_mlir_text(SOURCE), outline_literal_fills=value)
