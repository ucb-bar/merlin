"""Logical matrix subviews and ordered readouts remain source-only evidence."""

from __future__ import annotations

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.kernels.mechanism_inference import source_mechanisms
from merlin.kernels.source_product_views import source_product_views


def product(index, lhs, rhs):
    # Same typed source grammar as ordinary component-program output; independent
    # dimensions and generic identities keep this a source-view regression.
    return f"""
    %empty{index} = tensor.empty() : tensor<3x5xi32>
    %zero{index} = arith.constant 0 : i32
    %init{index} = linalg.fill ins(%zero{index} : i32) outs(%empty{index} : tensor<3x5xi32>) -> tensor<3x5xi32>
    %p{index} = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d2)>,
      affine_map<(d0,d1,d2) -> (d2,d1)>, affine_map<(d0,d1,d2) -> (d0,d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%{lhs}, %{rhs} : tensor<3x4xi8>, tensor<4x5xi8>) outs(%init{index} : tensor<3x5xi32>) {{
      ^bb{index}(%aa{index}:i8, %bb{index}:i8, %c{index}:i32):
        %x{index} = arith.extsi %aa{index} : i8 to i32
        %y{index} = arith.extsi %bb{index} : i8 to i32
        %m{index} = arith.muli %x{index}, %y{index} : i32
        %s{index} = arith.addi %c{index}, %m{index} : i32
        linalg.yield %s{index} : i32
    }} -> tensor<3x5xi32>
    """


def source():
    return """module { func.func @example(%a: tensor<3x12xi8>, %b: tensor<12x5xi8>) -> tensor<6x5xi32> {
      %a0 = tensor.extract_slice %a[0,1][3,4][1,1] : tensor<3x12xi8> to tensor<3x4xi8>
      %b0 = tensor.extract_slice %b[2,0][4,5][1,1] : tensor<12x5xi8> to tensor<4x5xi8>
      %ap = tensor.extract_slice %a[0,4][3,8][1,1] : tensor<3x12xi8> to tensor<3x8xi8>
      %a1 = tensor.extract_slice %ap[0,2][3,4][1,1] : tensor<3x8xi8> to tensor<3x4xi8>
      %b1 = tensor.extract_slice %b[6,0][4,5][1,1] : tensor<12x5xi8> to tensor<4x5xi8>
    """ + product(0, "a0", "b0") + product(1, "a1", "b1") + """
      %joined = tensor.concat dim(0) %p0, %p1 : (tensor<3x5xi32>,tensor<3x5xi32>) -> tensor<6x5xi32>
      return %joined : tensor<6x5xi32>
    } }"""


def observe(text=None, **kwargs):
    module = parse_mlir_text(source() if text is None else text)
    module.verify()
    return source_product_views(module, evidence_id="actual-source", **kwargs)


def test_nested_slices_retain_root_arguments_nonzero_reduction_offsets_and_concat_order():
    report = observe()
    assert report["status"] == "observed" and report["unknowns"] == []
    p0, p1 = report["products"]
    assert p0["lhs"]["argument"] == 0 and p0["lhs"]["offsets"] == [0, 1]
    assert p0["rhs"]["argument"] == 1 and p0["rhs"]["offsets"] == [2, 0]
    assert p1["lhs"]["offsets"] == [0, 6] and p1["rhs"]["offsets"] == [6, 0]
    assert p1["reduction"] == dict(extent=4, lhs_axis=1, rhs_axis=0)
    assert p1["numeric"]["operand_extensions"] == ["signed", "signed"]
    assert report["outputs"][0]["placements"] == [dict(product=0, offsets=[0, 0], shape=[3, 5]),
                                                  dict(product=1, offsets=[3, 0], shape=[3, 5])]
    assert report["coverage"] == dict(source_products_total=2, source_products_bound=2,
        ordered_returns_total=1, ordered_returns_bound=1, products_omitted=0, outputs_omitted=0)
    assert report["source_capture_equivalence"] == report["source_to_command_buffer"] == "unknown"
    assert report["numerical_equivalence"] == report["emitted_application"] == "unknown"


def test_nonunit_positive_stride_is_composed_as_a_logical_view_not_physical_packing():
    text = source().replace("%a[0,1][3,4][1,1]", "%a[0,1][3,4][1,2]")
    report = observe(text)
    assert report["status"] == "observed"
    assert report["products"][0]["lhs"]["strides"] == [1, 2]
    assert report["coordinate_scope"] == "logical_function_arguments_and_ordered_returns"


@pytest.mark.parametrize("slice_text", ["%a[0,10][3,4][1,1]", "%a[0,1][3,4][1,0]", "%a[0,-1][3,4][1,1]"])
def test_out_of_bounds_negative_offset_and_zero_stride_cannot_supply_a_view(slice_text):
    report = observe(source().replace("%a[0,1][3,4][1,1]", slice_text))
    assert report["status"] == "partial"
    assert "slice_offsets_sizes_or_strides_unproved" in {row["reason"] for row in report["unknowns"]}
    assert report["coverage"]["source_products_bound"] < report["coverage"]["source_products_total"]


def test_concat_axis_and_input_order_are_observed_in_output_coordinates():
    ordered = observe()["outputs"][0]["placements"]
    reversed_report = observe(source().replace("dim(0) %p0, %p1", "dim(0) %p1, %p0"))
    assert reversed_report["outputs"][0]["placements"] != ordered
    assert reversed_report["outputs"][0]["placements"][0]["product"] == 1
    columns = source().replace("tensor<6x5xi32>", "tensor<3x10xi32>").replace("dim(0)", "dim(1)")
    report = observe(columns)
    assert report["status"] == "observed"
    assert report["outputs"][0]["placements"][1]["offsets"] == [0, 5]


def test_repeated_product_use_and_return_fanout_remain_explicit():
    text = source().replace("dim(0) %p0, %p1", "dim(0) %p0, %p0")
    report = observe(text)
    assert report["status"] == "partial"  # The second source product is not a returned product.
    assert [row["readout_uses"] for row in report["products"]] == [2, 0]
    text = source().replace("-> tensor<6x5xi32> {", "-> (tensor<6x5xi32>, tensor<6x5xi32>) {")
    text = text.replace("return %joined : tensor<6x5xi32>",
                        "return %joined, %joined : tensor<6x5xi32>, tensor<6x5xi32>")
    report = observe(text)
    assert report["status"] == "observed" and report["outputs_total"] == 2
    assert [row["readout_uses"] for row in report["products"]] == [2, 2]


@pytest.mark.parametrize("before,after,reason", [
    ("arith.constant 0", "arith.constant 1", "zero_initialization_unproved"),
    ("arith.addi %c0, %m0", "arith.subi %c0, %m0", "complete_matrix_product_domain_unproved"),
    ("arith.muli %x0, %y0 : i32", "arith.muli %x0, %y0 overflow<nsw> : i32",
     "overflow_flagged_arithmetic_requires_numeric_adapter"),
])
def test_numeric_changes_do_not_inherit_the_modular_zero_initialized_product(before, after, reason):
    report = observe(source().replace(before, after))
    assert report["status"] == "partial"
    assert reason in {row["reason"] for row in report["unknowns"]}
    assert report["source_capture_equivalence"] == "unknown"


def test_dynamic_slice_and_arbitrary_tensor_readout_preserve_partial_coverage():
    text = source().replace("%b: tensor<12x5xi8>)", "%b: tensor<12x5xi8>, %offset: index)")
    text = text.replace("%a[0,1][3,4][1,1]", "%a[0,%offset][3,4][1,1]")
    report = observe(text)
    assert report["status"] == "partial"
    assert "slice_requires_bound_static_input_view" in {row["reason"] for row in report["unknowns"]}
    text = source().replace("return %joined : tensor<6x5xi32>",
        "%changed = arith.addi %joined, %joined : tensor<6x5xi32>\nreturn %changed : tensor<6x5xi32>")
    report = observe(text)
    assert report["status"] == "partial" and report["coverage"]["ordered_returns_bound"] == 0
    assert report["products_total"] == 2 and report["source_capture_equivalence"] == "unknown"


def test_budget_exhaustion_has_no_complete_prefix_and_source_diagnostic_integration_is_explicit():
    assert observe(max_products=1) == dict(status="unknown", reason="source_product_view_product_budget_exhausted")
    assert observe(max_placements=1) == dict(status="unknown", reason="source_product_view_placement_budget_exhausted")
    module = parse_mlir_text(source())
    module.verify()
    report = source_mechanisms(module, evidence_id="parsed", max_witnesses=1)
    assert report["reductions_omitted"] == 1
    assert report["product_views"]["products_total"] == 2
    assert report["product_views"]["coverage"]["products_omitted"] == 0


def test_repeated_concat_operands_stop_before_expanding_past_the_placement_budget(monkeypatch):
    from merlin.kernels import source_product_views as adapter

    count = 1024
    text = source().replace("tensor<6x5xi32>", f"tensor<{3 * count}x5xi32>")
    text = text.replace("dim(0) %p0, %p1 :", "dim(0) " + ", ".join(["%p0"] * count) + " :")
    text = text.replace("(tensor<3x5xi32>,tensor<3x5xi32>)", "(" + ",".join(["tensor<3x5xi32>"] * count) + ")")
    module = parse_mlir_text(text)
    module.verify()
    product = next(op for op in module.walk() if op.name == "linalg.generic").results[0]
    matrix, inspected = adapter._matrix, []

    def bounded(value):
        if value is product:
            inspected.append(value)
            assert len(inspected) <= 3, "one product plus at most two admitted concat operands"
        return matrix(value)

    monkeypatch.setattr(adapter, "_matrix", bounded)
    report = source_product_views(module, evidence_id="bounded", max_placements=2)
    assert report == dict(status="unknown", reason="source_product_view_placement_budget_exhausted")
    assert len(inspected) == 3


def test_ordered_return_fanout_cannot_expand_past_the_total_placement_budget():
    text = source().replace("-> tensor<6x5xi32> {", "-> (tensor<6x5xi32>, tensor<6x5xi32>) {")
    text = text.replace("return %joined : tensor<6x5xi32>",
                        "return %joined, %joined : tensor<6x5xi32>, tensor<6x5xi32>")
    assert observe(text, max_placements=3) == dict(
        status="unknown", reason="source_product_view_placement_budget_exhausted")
