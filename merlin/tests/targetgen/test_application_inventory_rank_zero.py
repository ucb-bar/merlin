"""Scalar tensors have observed shapes even when their ATen source was a reduction."""

import pytest

from merlin.targetgen.application_inventory import application_demand_inventory


@pytest.mark.parametrize("shape", ["", "2x3x"])
@pytest.mark.parametrize(
    "frontend, body, dtype, result_dtype",
    [
        ("aten.div.Scalar", "arith.divf", "f32", "f32"),
        ("aten.mul.Scalar", "arith.mulf", "f32", "f32"),
        ("aten.mul.Scalar", "arith.sitofp", "i8", "f32"),
        ("aten.sub.Scalar", "arith.subf", "bf16", "bf16"),
        ("aten.sub.Tensor", "arith.subi", "i64", "i64"),
        ("aten.mean.default", "arith.divf", "f32", "f32"),
        ("aten.sum.dim_IntList", "arith.truncf", "f32", "f16"),
    ],
)
def test_observed_scalar_tensor_helpers_follow_hardware_eligibility(
    tmp_path, shape, frontend, body, dtype, result_dtype
):
    source, result = f"tensor<{shape}{dtype}>", f"tensor<{shape}{result_dtype}>"
    dims = "d0, d1" if shape else ""
    indexing = f"affine_map<({dims}) -> ({dims})>"
    iterators = ", ".join(["#linalg.iterator_type<parallel>"] * (2 if shape else 0))
    expression = (
        f'"{body}"(%a) : ({dtype}) -> {result_dtype}'
        if body in {"arith.truncf", "arith.sitofp"}
        else f'"{body}"(%a, %a) : ({dtype}, {dtype}) -> {result_dtype}'
    )
    capture = tmp_path / "model.mlir"
    capture.write_text(f'''builtin.module {{
      func.func @forward(%x: {source}, %out: {result}) -> {result} {{
        %r = "linalg.generic"(%x, %out) <{{
          indexing_maps = [{indexing}, {indexing}],
          iterator_types = [{iterators}], operandSegmentSizes = array<i32: 1, 1>
        }}> ({{
          ^bb0(%a: {dtype}, %b: {result_dtype}):
            %v = {expression}
            "linalg.yield"(%v) : ({result_dtype}) -> ()
        }}) {{prov.aten = "{frontend}", prov.family = "reduce"}}
          : ({source}, {result}) -> {result}
        func.return %r : {result}
      }}
    }}''')
    # An int8-only accelerator cannot execute these typed standalone maps.
    # No host/spec roster is needed to observe that host lowering is required.
    report = application_demand_inventory(
        {"probe": capture},
        "test_device",
        detailed=True,
        capability_contract={
            "name": "test_device",
            "compute_units": [
                {
                    "name": "array",
                    "kind": "systolic",
                    "semantic_capabilities": [
                        {
                            "family": "elementwise_map",
                            "dtypes": ["int8"],
                            "result_dtypes": ["int8", "int32"],
                        }
                    ],
                }
            ],
        },
    )
    assert report["status"] == "inventoried"
    app = report["applications"]["probe"]
    (helper,) = [row for row in app["signatures"] if row["mlir_operation"] == "linalg.generic"]
    assert helper["operation"] == frontend
    assert helper["semantic_family"] == "elementwise_map"
    assert helper["shape_confidence"] == "result_type"
    assert helper["result_shapes"] == ([[2, 3]] if shape else [[]])
    assert helper["disposition"] == "host_required"
    assert not app["counts"].get("hardware_admitted")
    assert sum(row["count"] for row in app["signatures"]) == app["n_operations"]


def test_external_call_with_scalar_tensor_result_still_needs_a_declaration(tmp_path):
    capture = tmp_path / "model.mlir"
    capture.write_text("""builtin.module {
      func.func private @external(tensor<f32>) -> tensor<f32>
      func.func @forward(%x: tensor<f32>) -> tensor<f32> {
        %r = func.call @external(%x) : (tensor<f32>) -> tensor<f32>
        func.return %r : tensor<f32>
      }
    }""")
    report = application_demand_inventory(
        {"probe": capture},
        "test_device",
        detailed=True,
        capability_contract={"name": "test_device", "compute_units": []},
    )
    assert report["status"] == "incomplete"
    (call,) = [row for row in report["applications"]["probe"]["signatures"] if row["mlir_operation"] == "func.call"]
    assert call["result_shapes"] == [[]]
    assert call["disposition"] == "unclassified"
