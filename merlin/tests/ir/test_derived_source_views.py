"""Original model SSA, scalar fields and executable source pointers must agree."""

import pytest
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.dialects.func import CallOp

from merlin.frontends.derived_runtime_arguments import checked_runtime_call_shapes
from merlin.frontends.derived_source_views import checked_source_views, derive
from merlin.frontends.linalg_mlir import parse_mlir_text


def example():
    module = parse_mlir_text('''module {
      func.func @forward(%root:tensor<6x16xi64>,%other:tensor<6x16xi64>)->(tensor<6x16xi64>,tensor<8x6xi64>) {
        %empty=tensor.empty():tensor<16x6xi64>
        %t=linalg.transpose ins(%root:tensor<6x16xi64>) outs(%empty:tensor<16x6xi64>) permutation=[1,0]
        %view="tensor.extract_slice"(%t) <{static_offsets=array<i64:1,0>,static_sizes=array<i64:8,6>,
          static_strides=array<i64:2,1>,operandSegmentSizes=array<i32:1,0,0,0>}>:(tensor<16x6xi64>)->tensor<8x6xi64>
        %r:2=func.call @kernel(%root,%view) {prov.kernel="test::view",prov.kernel_operands=["owner","view"],
          prov.kernel_args={rows=8:i64,cols=6:i64,s0=2:i64,s1=16:i64,origin=1:i64}}
          :(tensor<6x16xi64>,tensor<8x6xi64>)->(tensor<6x16xi64>,tensor<8x6xi64>)
        return %r#0,%r#1:tensor<6x16xi64>,tensor<8x6xi64>
      }
      func.func private @kernel(tensor<6x16xi64>,tensor<8x6xi64>)->(tensor<6x16xi64>,tensor<8x6xi64>)
        attributes {prov.kernel="test::view"}
    }''')
    module.verify()
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    root, view = call.arguments
    names = ("rows", "cols", "s0", "s1", "origin")
    values = dict(zip(names, (8, 6, 2, 16, 1), strict=True))
    mapping, by_name = {"state": "owner"}, {"owner": root, "view": view}
    config = {"root": "state", "source_arg": 1, "shape": ["rows", "cols"], "strides": ["s0", "s1"], "offset": "origin"}
    records = {"view": {"binding": config, "root_schema_operand": "owner", "root_type": str(root.type),
        "view_type": str(view.type), "layout": {"shape": [8, 6], "strides": [2, 16], "offset": 1, "steps": [
            {"op": "linalg.transpose", "permutation": [1, 0]},
            {"op": "tensor.extract_slice", "offsets": [1, 0], "sizes": [8, 6], "strides": [2, 1]}]}}}
    provenance = {"source_argument_binding": {"bindings": [{"kind": "pointer", "allocation": "state", "offset": 0},
        {"kind": "pointer", "allocation": "state", "offset": "origin"}]}}
    spec = {"schema": "t2m.shared-tensor-abi.v3", "buffers": [{"name": "state", "shape": [6, 16], "dtype": "i64", "effect": "readwrite"},
        {"name": "out", "shape": [8, 6], "dtype": "i64", "effect": "write"}],
        "runtime_scalars": [{"name": name, "dtype": "i32", "source_arg": i + 2} for i, name in enumerate(names)]}
    record = {"kernel": "test::view", "callee": "kernel", "parent_call_ordinal": 0,
        "schema_operands": ["owner", "view"], "source_operand_bindings": mapping, "source_result_order": ["state", "out"],
        "model_call_attributes": {key: str(value) for key, value in call.attributes.items()},
        "runtime_scalar_bindings": {name: {"schema_argument": name, "attribute": str(IntegerAttr(values[name], i64))} for name in names},
        "source_tensor_views": records}
    return module, call, mapping, by_name, values, records, provenance, spec, record


def test_consumer_recovers_coordinates_and_checks_actual_original_call():
    module, call, mapping, by_name, values, records, provenance, spec, record = example()
    assert derive(call.arguments[1], call.arguments[0]) == records["view"]["layout"]
    checked_source_views(records, mapping, by_name, values, provenance)
    assert checked_runtime_call_shapes(module, spec, record, source_views=True, provenance=provenance) == {
        "state": [6, 16], "out": [8, 6]}


@pytest.mark.parametrize("change", ("offset", "stride", "shape", "root", "source-pointer", "source-arg", "type",
    "layout", "steps", "missing", "extra", "scalar-type", "transpose-body", "scalar-binding", "profile", "call"))
def test_rehashed_claims_do_not_replace_parent_dataflow_or_source_bindings(change):
    module, call, mapping, by_name, values, records, provenance, spec, record = example()
    if change == "offset": values["origin"] = 0
    elif change == "stride": values["s0"] = 1
    elif change == "shape": values["rows"] = 7
    elif change == "root": by_name["owner"] = call.parent_block().args[1]
    elif change == "source-pointer": provenance["source_argument_binding"]["bindings"][1]["allocation"] = "out"
    elif change == "source-arg": records["view"]["binding"]["source_arg"] = 0
    elif change == "type": records["view"]["root_type"] = "tensor<96xi64>"
    elif change == "layout": records["view"]["layout"]["offset"] = 0
    elif change == "steps": records["view"]["layout"]["steps"] = []
    elif change == "missing": records["view"].pop("layout")
    elif change == "extra": records["view"]["binding"]["grants_alias"] = True
    elif change == "scalar-type": values["origin"] = True
    elif change == "transpose-body":
        body = call.arguments[1].owner.operands[0].owner.regions[0].block
        body.last_op.operands = (body.args[-1],)
    elif change == "scalar-binding": record["runtime_scalar_bindings"]["origin"]["attribute"] = "0 : i64"
    elif change == "call": record["model_call_attributes"] = {}
    before = str(module)
    with pytest.raises(ValueError):
        if change in ("scalar-binding", "profile", "call"):
            checked_runtime_call_shapes(module, spec, record, source_views=change != "profile", provenance=provenance)
        else:
            checked_source_views(records, mapping, by_name, values, provenance)
    assert str(module) == before
