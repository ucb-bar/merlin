"""Closed scalar extract/rebox fragments retain original compute-only bodies."""

import pytest
from xdsl.dialects.builtin import StringAttr, TensorType, i64

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import plan_deployable_chain
from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

SOURCE = """module {func.func @forward(%x:tensor<i64>)->tensor<1xi64>{
 %v=tensor.extract %x[]:tensor<i64>
 %y=tensor.from_elements %v:tensor<1xi64>
 %e=tensor.empty():tensor<1xi64>
 %r=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
 ins(%y:tensor<1xi64>)outs(%e:tensor<1xi64>){
 ^bb0(%a:i64,%b:i64):
 %one=arith.constant 1:i64
 %v1=arith.addi %a,%one:i64
 linalg.yield %v1:i64
 }->tensor<1xi64>
 func.return %r:tensor<1xi64>
}}
"""


def test_exact_rebox_preserves_original_operations_and_tensor_boundaries():
    original = parse_mlir_text(SOURCE)
    before = str(original)
    planned = plan_deployable_chain(
        original,
        worker_slots=(0, 1),
        worker_count=2,
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        data_preparation=True,
    )
    assert str(original) == before
    assert len(planned.data_preparation["scalar_tensor_view_kernels"]) == 1
    first = planned.dispatches[0]
    kernel = next(
        op
        for op in planned.outlined.module.body.block.ops
        if op.name == "func.func" and op.sym_name.data == first["symbol"]
    )
    assert [str(t) for t in kernel.function_type.inputs] == ["tensor<i64>"]
    assert [str(t) for t in kernel.function_type.outputs] == ["tensor<1xi64>"]
    assert [op.name for op in kernel.body.block.ops] == ["tensor.extract", "tensor.from_elements", "func.return"]
    assert planned.to_dict()["controller_arithmetic"] is False


def test_default_preparation_does_not_reinterpret_a_scalar_driver_value():
    with pytest.raises(ValueError, match="ranked tensor"):
        plan_deployable_chain(
            parse_mlir_text(SOURCE),
            worker_slots=(0,),
            worker_count=1,
            workspace_budget_bytes=4096,
            alignment_bytes=64,
        )


@pytest.mark.parametrize("dtype", ["i8", "i16", "i32", "i64", "i1", "f32", "f64"])
def test_supported_element_storage_keeps_exact_type_and_body(dtype):
    module = parse_mlir_text(
        "module {func.func @forward(%x:tensor<" + dtype + ">)->tensor<1x" + dtype + ">{"
        "%v=tensor.extract %x[]:tensor<" + dtype + ">\n"
        "%y=tensor.from_elements %v:tensor<1x" + dtype + ">\n"
        "func.return %y:tensor<1x" + dtype + ">}}"
    )
    outlined = outline_dispatches(module, outline_scalar_views=True)
    assert len(outlined.dispatches) == 1
    outlined.module.verify()


@pytest.mark.parametrize("change", ["unknown", "quant_inner", "different_origin", "extra_use", "encoding", "rank"])
def test_unproved_attribute_encoding_or_additional_scalar_use_retains_normal_refusal(change):
    module = parse_mlir_text(SOURCE)
    extracted = next(op for op in module.walk() if op.name == "tensor.extract")
    boxed = next(op for op in module.walk() if op.name == "tensor.from_elements")
    if change in ("unknown", "quant_inner", "different_origin"):
        key = {"unknown": "test.effect", "quant_inner": "prov.quant_inner", "different_origin": "prov.origin"}[change]
        boxed.attributes[key] = StringAttr("unproved")
    elif change == "encoding":
        boxed.results[0]._type = TensorType(i64, [1], StringAttr("unknown"))
    elif change == "rank":
        boxed.results[0]._type = TensorType(i64, [])
    else:
        from xdsl.dialects import tensor
        from xdsl.rewriter import InsertPoint, Rewriter

        other = tensor.FromElementsOp(extracted.results[0])
        Rewriter.insert_op(other, InsertPoint.before(boxed))
    before = str(module)
    result = outline_dispatches(module, outline_scalar_views=True)
    assert [d.root_op for d in result.dispatches] == ["linalg.generic"]
    assert str(module) == before


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_scalar_view_selection_requires_exact_boolean(value):
    with pytest.raises(OutlineError, match="explicit boolean"):
        outline_dispatches(parse_mlir_text(SOURCE), outline_scalar_views=value)
