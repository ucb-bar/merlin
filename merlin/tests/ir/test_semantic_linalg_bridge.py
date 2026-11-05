"""Parsed Linalg-to-native graph boundary checks for the admitted integer subset."""

from __future__ import annotations

import os
import random
import subprocess
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from struct import pack, unpack

import pytest

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root
from merlin.frontends.linalg_patterns import InvalidLinalgPattern, recognize_static_projected_pointwise
from merlin.semantic_compiler.allocate import StorageBank, check_assignment
from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, _f32_div_rne, evaluate_graph
from merlin.semantic_compiler.rules import (
    AxisBound,
    AxisEquality,
    ConcatViewDescriptor,
    InstructionDescriptor,
    ViewDescriptor,
    generate_rules,
)
from merlin.semantic_compiler.search import select_and_allocate
from merlin.semantic_compiler.snapshot import NativeTargetProfile, build_native_snapshot, open_native_snapshot
from merlin.semantic_compiler.verify import check_selection


@pytest.fixture(scope="module")
def bridge(tmp_path_factory: pytest.TempPathFactory) -> Path:
    manifest = repo_root() / "src/merlin/semantic_compiler/egg_bridge/Cargo.toml"
    target = tmp_path_factory.mktemp("linalg-native-egg")
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest)], check=True, env=environment
    )
    return target / "release/merlin-egg-bridge"


def _generic_source(
    *, init: str = "argument", body: str = "signed", swapped_map: bool = False, two_outputs: bool = False
) -> str:
    function_args = "%a: tensor<2x3xi8>, %b: tensor<3x2xi8>"
    prefix = ""
    if init == "argument":
        function_args += ", %c: tensor<2x2xi32>"
    else:
        prefix = "%empty = tensor.empty() : tensor<2x2xi32>\n"
        if init == "filled":
            prefix += (
                "%seven = arith.constant 7 : i32\n"
                "%c = linalg.fill ins(%seven : i32) outs(%empty : tensor<2x2xi32>) -> tensor<2x2xi32>\n"
            )
        else:
            prefix += "%c = tensor.empty() : tensor<2x2xi32>\n"
    left_map = "(k,m)" if swapped_map else "(m,k)"
    cast = "extui" if body == "unsigned" else "extsi"
    if body == "dropped_accumulator":
        yield_operand = "%p"
    else:
        yield_operand = "%v"
    result_types = "(tensor<2x2xi32>, tensor<2x2xi32>)" if two_outputs else "tensor<2x2xi32>"
    returned = "%r, %c : tensor<2x2xi32>, tensor<2x2xi32>" if two_outputs else "%r : tensor<2x2xi32>"
    return f"""module {{
func.func @work({function_args}) -> {result_types} {{
{prefix}%r = linalg.generic {{indexing_maps = [affine_map<(m,n,k)->{left_map}>,
  affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
  iterator_types = ["parallel", "parallel", "reduction"]}}
  ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%c : tensor<2x2xi32>) {{
  ^bb0(%x: i8, %y: i8, %acc: i32):
  %xx = arith.{cast} %x : i8 to i32
  %yy = arith.extsi %y : i8 to i32
  %p = arith.muli %xx, %yy : i32
  %v = arith.addi %p, %acc : i32
  linalg.yield {yield_operand} : i32
  }} -> tensor<2x2xi32>
  func.return {returned}
}}
}}"""


def _inputs(*, init: int = 10) -> dict[str, TensorValue]:
    return {
        "arg0": TensorValue(TensorType((2, 3), "i8", "signed-i8"), (1, 2, 3, 4, 5, 6)),
        "arg1": TensorValue(TensorType((3, 2), "i8", "signed-i8"), (1, 2, 3, 4, 5, 6)),
        "arg2": TensorValue(TensorType((2, 2), "i32", "i32-wrap-k-ascending"), (init,) * 4),
    }


def test_lifted_signed_weight_preserves_bytes_and_ordered_outputs(bridge: Path):
    source = """module {
func.func @work(%weight: tensor<2xi8>, %runtime: tensor<2xi8>)
  -> (tensor<2xi8>, tensor<2xi8>) {
  func.return %weight, %runtime : tensor<2xi8>, tensor<2xi8>
}
}"""
    translated = translate_linalg_text(
        source, entry="work", target_identity="synthetic", argument_constants={0: b"\xff\x02"},
    )
    request = translated.request
    assert request.input_storages == (("arg1", "external"),)
    assert [(item.node_id, item.encoding, item.data_hex) for item in request.constants] == [
        ("arg0", "i8", "ff02"),
    ]
    assert evaluate_graph(
        request,
        {"arg1": TensorValue(request.node("arg1").type, (3, 4))},
        constants={"arg0": TensorValue(request.node("arg0").type, (-1, 2))},
    ) == (
        TensorValue(request.node("arg0").type, (-1, 2)),
        TensorValue(request.node("arg1").type, (3, 4)),
    )
    selected = select_and_allocate(
        request, (), (StorageBank("external", "dram", 4, "word"),),
        bridge=bridge, fixed_inputs={"arg1": 0},
    )
    assert selected.status == "selected", selected.reason
    assert selected.allocation is not None
    assert set(selected.allocation.addresses.values()) == {0, 1}
    with pytest.raises(LinalgBridgeError, match="differs from its parsed tensor type"):
        translate_linalg_text(
            source, entry="work", target_identity="synthetic", argument_constants={0: b"\xff"},
        )
    with pytest.raises(LinalgBridgeError, match="invalid indices or bytes"):
        translate_linalg_text(
            source, entry="work", target_identity="synthetic", argument_constants={2: b"\xff\x02"},
        )


def _static_insert_source(*, destination: str = "%dest", sizes: str = "2, 2",
                          offsets: str = "0, 1", strides: str = "1, 2") -> str:
    prelude = "%dest = tensor.empty() : tensor<3x4xf32>" if destination == "%dest" else ""
    args = "%source: tensor<2x2xf32>" if prelude else (
        "%source: tensor<2x2xf32>, %destination: tensor<3x4xf32>"
    )
    actual_destination = "%dest" if prelude else "%destination"
    return f'''module {{
func.func @work({args}) -> tensor<3x4xf32> {{
  {prelude}
  %result = "tensor.insert_slice"(%source, {actual_destination}) <{{
    static_offsets = array<i64: {offsets}>, static_sizes = array<i64: {sizes}>,
    static_strides = array<i64: {strides}>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>
  }}> : (tensor<2x2xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %result : tensor<3x4xf32>
}}
}}'''


def _static_extract_source(*, offsets: str = "0, 1", sizes: str = "2, 2",
                           strides: str = "1, 2", initialized: bool = True) -> str:
    prelude = "" if initialized else "%source = tensor.empty() : tensor<2x5xf32>"
    arguments = "" if prelude else "%input: tensor<2x5xf32>"
    actual_source = "%source" if prelude else "%input"
    return f'''module {{
func.func @work({arguments}) -> tensor<2x2xf32> {{
  {prelude}
  %result = "tensor.extract_slice"({actual_source}) <{{
    static_offsets = array<i64: {offsets}>, static_sizes = array<i64: {sizes}>,
    static_strides = array<i64: {strides}>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
  }}> : (tensor<2x5xf32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}}
}}'''


def test_static_extract_slice_preserves_f32_bits_and_strided_indices():
    translated = translate_linalg_text(
        _static_extract_source(), entry="work", target_identity="synthetic",
    )
    request = translated.request
    extracted = request.nodes[-1]
    assert extracted.op == "gather" and extracted.inputs == ("arg0",)
    assert extracted.index_maps[0].apply((1, 1)) == (1, 3)
    source = TensorValue(request.node("arg0").type, (
        0, 0x80000000, 2, 0x7FC00001, 4,
        5, 0x3F800000, 7, 0xFF800000, 9,
    ))
    result, = evaluate_graph(request, {"arg0": source})
    assert result.elements == (0x80000000, 0x7FC00001, 0x3F800000, 0xFF800000)


def test_static_extract_slice_reaches_generated_native_rule(bridge: Path):
    request = translate_linalg_text(
        _static_extract_source(), entry="work", target_identity="synthetic-slice-v1",
    ).request
    node = request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_strided_gather", "gather", ("external",), "external",
        "f32", "raw-bit-pattern", (2,),
        input_dtypes=("f32",), input_numerical_policies=("raw-bit-pattern",),
        input_ranks=(2,), index_maps=node.index_maps, shape_contract="bounded",
        output_axis_bounds=(AxisBound(0, 2, 2), AxisBound(1, 2, 2)),
        input_axis_bounds=((AxisBound(0, 2, 2), AxisBound(1, 5, 5)),),
    )
    selected = select_and_allocate(
        request, (descriptor,), (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0}, fixed_outputs=(1,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    assert not descriptor.accepts(replace(node, index_maps=tuple(reversed(node.index_maps))),
                                  (request.nodes[0],))


def test_contiguous_extract_slice_uses_checked_offset_view(bridge: Path) -> None:
    source = '''module {
func.func @work(%source: tensor<6xf32>) -> tensor<2xf32> {
  %result = "tensor.extract_slice"(%source) <{
    static_offsets = array<i64: 2>, static_sizes = array<i64: 2>,
    static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
  }> : (tensor<6xf32>) -> tensor<2xf32>
  func.return %result : tensor<2xf32>
} }'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-slice-view")
    request = replace(
        translated.request, physical_extents=(("arg0", 6),),
        boundary_layouts=(("arg0", "contiguous_element_order"),),
        output_layouts=("contiguous_element_order",),
    )
    source_type, result_type = request.nodes[0].type, request.nodes[1].type
    view = ViewDescriptor(
        "slice_two", "external", source_type, result_type, 2,
        kind="slice", source_extent=6, physical_offset=2,
    )
    assert ViewDescriptor.from_record(view.record()) == view
    banks = (StorageBank("external", "dram", 6, "element"),)
    selected = select_and_allocate(
        request, (), banks, bridge=bridge, views=(view,),
        fixed_inputs={"arg0": 0}, fixed_outputs=(2,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 0
    assert selected.graph is not None and selected.allocation is not None
    assert selected.allocation.addresses == {0: 0, 1: 2}
    assert check_assignment(selected.graph, (), {0: 0, 1: 1}, banks)[1] == (
        "slice view has wrong source offset"
    )
    input_value = TensorValue(source_type, (0, 1, 2, 3, 4, 5))
    assert evaluate_graph(request, {"arg0": input_value})[0].elements == (2, 3)
    wrong_map = replace(request.nodes[1], index_maps=(
        replace(request.nodes[1].index_maps[0], coefficients=((2,),)),
        request.nodes[1].index_maps[1],
    ))
    assert not generate_rules(replace(request, nodes=(request.nodes[0], wrong_map)), (), (view,)).rewrites
    assert not generate_rules(request, (), (replace(view, physical_offset=1),)).rewrites
    with pytest.raises(ValueError, match="offset or extent"):
        replace(view, physical_offset=5)
    byte_request = replace(request, physical_extents=(("arg0", 24),))
    byte_view = replace(view, source_extent=24, extent=8, physical_offset=8)
    byte_banks = (StorageBank("external", "dram", 24, "byte", alignment=4),)
    byte_selected = select_and_allocate(
        byte_request, (), byte_banks, bridge=bridge, views=(byte_view,),
        fixed_inputs={"arg0": 0}, fixed_outputs=(8,),
    )
    assert byte_selected.status == "selected", byte_selected.reason
    assert byte_selected.allocation is not None and byte_selected.allocation.addresses == {0: 0, 1: 8}
    with pytest.raises(ValueError, match="offset or extent"):
        replace(byte_view, physical_offset=9)


def test_complete_row_extract_slice_uses_checked_offset_view(bridge: Path) -> None:
    source = '''module { func.func @work(%source: tensor<4x3xf32>) -> tensor<2x3xf32> {
      %result = "tensor.extract_slice"(%source) <{
        static_offsets = array<i64: 1, 0>, static_sizes = array<i64: 2, 3>,
        static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
      }> : (tensor<4x3xf32>) -> tensor<2x3xf32>
      func.return %result : tensor<2x3xf32>
    } }'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-row-slice-view")
    request = replace(
        translated.request, physical_extents=(("arg0", 48),),
        boundary_layouts=(("arg0", "contiguous_element_order"),),
        output_layouts=("contiguous_element_order",),
    )
    view = ViewDescriptor(
        "two_rows", "external", request.nodes[0].type, request.nodes[1].type, 24,
        kind="slice", source_extent=48, physical_offset=12,
    )
    banks = (StorageBank("external", "dram", 48, "byte", alignment=4),)
    selected = select_and_allocate(
        request, (), banks, bridge=bridge, views=(view,),
        fixed_inputs={"arg0": 0}, fixed_outputs=(12,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.allocation is not None and selected.allocation.addresses == {0: 0, 1: 12}
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, tuple(range(12))),
    })
    assert result.elements == (3, 4, 5, 6, 7, 8)
    with pytest.raises(ValueError, match="offset or extent"):
        replace(view, physical_offset=4)
    column_source = source.replace("tensor<2x3xf32>", "tensor<4x2xf32>").replace(
        "array<i64: 1, 0>", "array<i64: 0, 1>"
    ).replace("array<i64: 2, 3>", "array<i64: 4, 2>")
    column_request = translate_linalg_text(
        column_source, entry="work", target_identity="synthetic-column-slice"
    ).request
    with pytest.raises(ValueError, match="contiguous physical geometry"):
        replace(view, result_type=column_request.nodes[1].type)


def test_static_extract_slice_preserves_three_ordered_layout_outputs():
    source = '''module {
func.func @work(%source: tensor<1x8x96xf32>)
    -> (tensor<1x8x32xf32>, tensor<1x8x32xf32>, tensor<1x8x32xf32>) {
  %first = "tensor.extract_slice"(%source) <{
    static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 32>,
    static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
  }> : (tensor<1x8x96xf32>) -> tensor<1x8x32xf32>
  %second = "tensor.extract_slice"(%source) <{
    static_offsets = array<i64: 0, 0, 32>, static_sizes = array<i64: 1, 8, 32>,
    static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
  }> : (tensor<1x8x96xf32>) -> tensor<1x8x32xf32>
  %third = "tensor.extract_slice"(%source) <{
    static_offsets = array<i64: 0, 0, 64>, static_sizes = array<i64: 1, 8, 32>,
    static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>
  }> : (tensor<1x8x96xf32>) -> tensor<1x8x32xf32>
  func.return %third, %first, %second : tensor<1x8x32xf32>, tensor<1x8x32xf32>, tensor<1x8x32xf32>
}
}'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    request = translated.request
    assert tuple(node.op for node in request.nodes) == ("input", "gather", "gather", "gather")
    assert request.outputs == ("op2r0", "op0r0", "op1r0")
    payload = TensorValue(request.node("arg0").type, tuple(range(8 * 96)))
    outputs = evaluate_graph(request, {"arg0": payload})
    assert tuple(output.elements[:3] for output in outputs) == ((64, 65, 66), (0, 1, 2), (32, 33, 34))
    assert tuple(output.elements[-1] for output in outputs) == (767, 703, 735)


def test_static_extract_slice_rejects_uninitialized_and_out_of_bounds_sources():
    with pytest.raises(LinalgBridgeError, match="uninitialized"):
        translate_linalg_text(
            _static_extract_source(initialized=False), entry="work", target_identity="synthetic",
        )
    with pytest.raises(LinalgBridgeError, match="exceeds source bounds"):
        translate_linalg_text(
            _static_extract_source(offsets="0, 3"),
            entry="work", target_identity="synthetic",
        )
    with pytest.raises(LinalgBridgeError, match="rank-reduced or mismatched result"):
        translate_linalg_text(
            _static_extract_source(sizes="2, 1"),
            entry="work", target_identity="synthetic",
        )


def test_static_insert_slice_preserves_destination_and_strided_placement():
    translated = translate_linalg_text(
        _static_insert_source(destination="%destination"), entry="work", target_identity="synthetic",
    )
    request = translated.request
    inserted = request.nodes[-1]
    assert inserted.op == "insert_slice" and inserted.inputs == ("arg0", "arg1")
    source = TensorValue(request.node("arg0").type, (0x80000000, 0, 0x3F800000, 0x40000000))
    destination = TensorValue(request.node("arg1").type, tuple(range(12)))
    expected = list(range(12))
    for index, value in zip((1, 3, 5, 7), source.elements, strict=True):
        expected[index] = value
    assert evaluate_graph(request, {"arg0": source, "arg1": destination}) == (
        TensorValue(request.node(inserted.id).type, tuple(expected)),
    )
    assert destination.elements == tuple(range(12))


def test_static_insert_slice_reproduces_initialized_four_dimensional_padding():
    source = '''module {
func.func @pad(%source: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %destination = tensor.splat %zero : tensor<1x1x4x4xf32>
  %result = "tensor.insert_slice"(%source, %destination) <{
    static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 1, 2, 2>,
    static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>
  }> : (tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  func.return %result : tensor<1x1x4x4xf32>
}
}'''
    translated = translate_linalg_text(source, entry="pad", target_identity="synthetic")
    request = translated.request
    values = TensorValue(request.node("arg0").type, (1, 2, 3, 4))
    result, = evaluate_graph(request, {"arg0": values}, constants=translated.constants)
    expected = [0] * 16
    for index, value in zip((5, 6, 9, 10), values.elements, strict=True):
        expected[index] = value
    assert result.elements == tuple(expected)


def test_f32_identity_generic_is_a_gather_but_arithmetic_still_refuses():
    source = '''module {
func.func @work(%source: tensor<2xf32>) -> tensor<2xf32> {
  %empty = tensor.empty() : tensor<2xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
    iterator_types = ["parallel"]
  } ins(%source : tensor<2xf32>) outs(%empty : tensor<2xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
  } -> tensor<2xf32>
  func.return %result : tensor<2xf32>
}
}'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    assert translated.request.nodes[-1].op == "gather"
    value = TensorValue(translated.request.node("arg0").type, (0x80000000, 0x3F800000))
    assert evaluate_graph(translated.request, {"arg0": value}) == (value,)
    arithmetic = source.replace(
        "linalg.yield %value : f32",
        "%sum = arith.addf %value, %old : f32\n      linalg.yield %sum : f32",
    )
    with pytest.raises(LinalgBridgeError, match="f32 linalg.generic needs an admitted semantic decomposition"):
        translate_linalg_text(arithmetic, entry="work", target_identity="synthetic")


def _f32_square_source() -> str:
    return '''module {
func.func @work(%source: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
  %empty = tensor.empty() : tensor<1x2x4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x2x4xf32>) outs(%empty : tensor<1x2x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %square = arith.mulf %value, %value : f32
      linalg.yield %square : f32
  } -> tensor<1x2x4xf32>
  func.return %result : tensor<1x2x4xf32>
}
}'''


def test_static_f32_square_preserves_rounded_mul_and_semantic_identity():
    translated = translate_linalg_text(_f32_square_source(), entry="work", target_identity="synthetic")
    request = translated.request
    square = request.nodes[-1]
    assert square.op == "f32_mul" and square.inputs == ("arg0", "arg0")
    assert dict(square.attrs) == {"numeric_semantics": "ieee754-f32-mul-rne-v1"}
    assert len(square.index_maps) == 3
    source = TensorValue(request.node("arg0").type, (
        0x80000000, 0x3F800000, 0x3F000000, 0x7F7FFFFF,
        0x00000001, 0x7F800000, 0xBF800000, 0x00800000,
    ))
    result, = evaluate_graph(request, {"arg0": source})
    assert result.elements == (
        0x00000000, 0x3F800000, 0x3E800000, 0x7F800000,
        0x00000000, 0x7F800000, 0x3F800000, 0x00000000,
    )
    assert square.semantic_key() != replace(square, attrs=()).semantic_key()
    with pytest.raises(ValueError, match="NaN payload"):
        evaluate_graph(request, {"arg0": replace(source, elements=(0x7FC00001,) * 8)})
    with pytest.raises(ValueError, match="semantic contract"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], replace(square, attrs=()),)),
                       {"arg0": source})


def test_static_f32_square_reaches_generated_native_rule(bridge: Path):
    request = translate_linalg_text(
        _f32_square_source(), entry="work", target_identity="synthetic-square-v1",
    ).request
    square = request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_strict_f32_mul", "f32_mul", ("external", "external"), "external",
        "f32", "raw-bit-pattern", (3,),
        input_dtypes=("f32", "f32"),
        input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
        input_ranks=(3, 3), index_maps=square.index_maps,
        required_attrs=square.attrs, shape_contract="equal",
    )
    selected = select_and_allocate(
        request, (descriptor,), (StorageBank("external", "dram", 2, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0}, fixed_outputs=(1,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    assert not descriptor.accepts(replace(square, attrs=()), (request.nodes[0], request.nodes[0]))


def _f32_binary_multiply_source() -> str:
    return '''module {
func.func @work(%lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %empty = tensor.empty() : tensor<2x3xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>,
                     affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<2x3xf32>, tensor<2x3xf32>)
    outs(%empty : tensor<2x3xf32>) {
    ^bb0(%a: f32, %b: f32, %old: f32):
      %product = arith.mulf %a, %b : f32
      linalg.yield %product : f32
  } -> tensor<2x3xf32>
  func.return %result : tensor<2x3xf32>
}
}'''


def test_static_f32_binary_multiply_uses_both_typed_inputs():
    translated = translate_linalg_text(
        _f32_binary_multiply_source(), entry="work", target_identity="synthetic",
    )
    request = translated.request
    node = request.nodes[-1]
    assert node.op == "f32_mul" and node.inputs == ("arg0", "arg1")
    inputs = {
        "arg0": TensorValue(request.node("arg0").type,
                            (0x3F000000, 0xBF800000, 0x80000000, 0x40000000, 0x00800000, 0x7F7FFFFF)),
        "arg1": TensorValue(request.node("arg1").type,
                            (0x40400000, 0xC0000000, 0x3F800000, 0x3F000000, 0x3F800000, 0x40000000)),
    }
    result, = evaluate_graph(request, inputs)
    assert result.elements == (
        0x3FC00000, 0x40000000, 0x80000000, 0x3F800000, 0x00800000, 0x7F800000,
    )


def test_static_f32_binary_multiply_rounds_both_midpoint_directions():
    request = translate_linalg_text(
        _f32_binary_multiply_source(), entry="work", target_identity="synthetic",
    ).request
    inputs = {
        "arg0": TensorValue(request.node("arg0").type, (0x3F800800,) * 6),
        "arg1": TensorValue(request.node("arg1").type,
                            (0x3F800800, 0x3F801800, 0x3F800800, 0x3F801800,
                             0x3F800800, 0x3F801800)),
    }
    result, = evaluate_graph(request, inputs)
    assert result.elements == (0x3F801000, 0x3F802002) * 3
    with pytest.raises(ValueError, match="NaN result"):
        evaluate_graph(request, {
            "arg0": replace(inputs["arg0"], elements=(0,) * 6),
            "arg1": replace(inputs["arg1"], elements=(0x7F800000,) * 6),
        })


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("%a, %b", "%a, %old"),
        ("arith.mulf %a, %b : f32",
         '"arith.mulf"(%a, %b) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32'),
    ],
)
def test_static_f32_binary_multiply_rejects_changed_scalar_body(old: str, new: str):
    with pytest.raises(LinalgBridgeError, match="f32 linalg.generic needs an admitted semantic decomposition"):
        translate_linalg_text(
            _f32_binary_multiply_source().replace(old, new), entry="work", target_identity="synthetic",
        )


def _f32_projected_binary_source(operation: str) -> str:
    return f'''module {{
func.func @work(%lhs: tensor<2x3xf32>, %bias: tensor<3xf32>) -> tensor<2x3xf32> {{
  %empty = tensor.empty() : tensor<2x3xf32>
  %result = linalg.generic {{
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(j)>,
                     affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]
  }} ins(%lhs, %bias : tensor<2x3xf32>, tensor<3xf32>)
    outs(%empty : tensor<2x3xf32>) {{
    ^bb0(%a: f32, %b: f32, %old: f32):
      %value = arith.{operation} %a, %b : f32
      linalg.yield %value : f32
  }} -> tensor<2x3xf32>
  func.return %result : tensor<2x3xf32>
}}
}}'''


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("addf", (0x3FC00000, 0x3FC00000, 0x40800000,
                  0x40900000, 0x40900000, 0x40E00000)),
        ("subf", (0x3F000000, 0x40200000, 0x40000000,
                  0x40600000, 0x40B00000, 0x40A00000)),
        ("mulf", (0x3F000000, 0xBF800000, 0x40400000,
                  0x40000000, 0xC0200000, 0x40C00000)),
        ("divf", (0x40000000, 0xC0800000, 0x40400000,
                  0x41000000, 0xC1200000, 0x40C00000)),
    ],
)
def test_projected_f32_arithmetic_preserves_bias_map_and_rounding(operation: str, expected: tuple[int, ...]):
    request = translate_linalg_text(
        _f32_projected_binary_source(operation), entry="work", target_identity="synthetic",
    ).request
    node = request.nodes[-1]
    assert node.op == f"f32_{operation[:-1]}" and node.inputs == ("arg0", "arg1")
    assert node.index_maps[1].apply((1, 2)) == (2,)
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type,
                            (0x3F800000, 0x40000000, 0x40400000,
                             0x40800000, 0x40A00000, 0x40C00000)),
        "arg1": TensorValue(request.node("arg1").type,
                            (0x3F000000, 0xBF000000, 0x3F800000)),
    })
    assert result.elements == expected


def test_projected_f32_division_rounds_subnormal_ties_and_preserves_signs():
    source = _f32_projected_binary_source("divf")
    parsed = next(mq.walk(mq.parse(source), "linalg.generic"))
    with pytest.raises(InvalidLinalgPattern, match="registered pure arith pointwise"):
        recognize_static_projected_pointwise(parsed)
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    node = request.nodes[-1]
    assert node.op == "f32_div"
    assert dict(node.attrs) == {"numeric_semantics": "ieee754-f32-div-rne-v1"}
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    lhs = request.node("arg0").type
    rhs = request.node("arg1").type
    result, = evaluate_graph(request, {
        "arg0": TensorValue(lhs, (0x3F800000, 0x80000000, 0x00000001,
                                  0xBF800000, 0xBF800000, 0x00000003)),
        "arg1": TensorValue(rhs, (0x40400000, 0xBF800000, 0x40000000)),
    })
    assert result.elements == (
        0x3EAAAAAB, 0x00000000, 0x00000000,
        0xBEAAAAAB, 0x3F800000, 0x00000002,
    )
    exceptional, = evaluate_graph(request, {
        "arg0": TensorValue(lhs, (0x7F7FFFFF, 0x80000000, 0x7F800000,
                                  0xFF7FFFFF, 0x00000000, 0xFF800000)),
        "arg1": TensorValue(rhs, (0x3F000000, 0x7F800000, 0x40000000)),
    })
    assert exceptional.elements == (
        0x7F800000, 0x80000000, 0x7F800000,
        0xFF800000, 0x00000000, 0xFF800000,
    )
    with pytest.raises(ValueError, match="NaN result"):
        evaluate_graph(request, {
            "arg0": TensorValue(lhs, (0,) * 6),
            "arg1": TensorValue(rhs, (0,) * 3),
        })
    for changed in (
        source.replace("arith.divf %a, %b", "arith.divf %b, %a"),
        source.replace("arith.divf %a, %b : f32", '"arith.divf"(%a, %b) '
                       '<{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32'),
    ):
        with pytest.raises(LinalgBridgeError, match="admitted semantic decomposition"):
            translate_linalg_text(changed, entry="work", target_identity="synthetic")


def test_f32_division_matches_independent_fraction_neighbors():
    """The reference divider agrees with exact rational nearest-neighbor checks."""
    randomizer = random.Random(17)

    def fraction(bits: int) -> Fraction:
        return Fraction.from_float(unpack("<f", pack("<I", bits))[0])

    for _ in range(200):
        left = randomizer.randrange(0x7F800000) | (randomizer.randrange(2) << 31)
        right = randomizer.randrange(1, 0x7F800000) | (randomizer.randrange(2) << 31)
        quotient = fraction(left) / fraction(right)
        sign = (left ^ right) & 0x80000000
        magnitude = abs(quotient)
        try:
            approximate = unpack("<I", pack("<f", float(magnitude)))[0]
        except OverflowError:
            approximate = 0x7F800000
        if approximate == 0x7F800000:
            overflow_midpoint = fraction(0x7F7FFFFF) + Fraction(2) ** 103
            nearest = 0x7F800000 if magnitude >= overflow_midpoint else 0x7F7FFFFF
        else:
            neighboring = {max(0, approximate - 1), approximate, min(0x7F7FFFFF, approximate + 1)}
            nearest = min(neighboring, key=lambda bits: (abs(magnitude - fraction(bits)), bits & 1))
        assert _f32_div_rne(left, right) == sign | nearest


def test_projected_f32_add_reaches_generated_native_rule(bridge: Path):
    request = translate_linalg_text(
        _f32_projected_binary_source("addf"), entry="work", target_identity="synthetic-bias-v1",
    ).request
    node = request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_projected_f32_add", "f32_add", ("external", "external"), "external",
        "f32", "raw-bit-pattern", (2,),
        input_dtypes=("f32", "f32"),
        input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
        input_ranks=(2, 1), index_maps=node.index_maps, required_attrs=node.attrs,
        shape_contract="relations", shape_equalities=(
            AxisEquality("out", 0, "in0", 0), AxisEquality("out", 1, "in0", 1),
            AxisEquality("out", 1, "in1", 0),
        ),
    )
    selected = select_and_allocate(
        request, (descriptor,), (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0, "arg1": 1}, fixed_outputs=(2,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    assert not descriptor.accepts(replace(node, op="f32_mul"), request.nodes[:2])


def test_projected_f32_reference_rejects_a_changed_input_map():
    request = translate_linalg_text(
        _f32_projected_binary_source("addf"), entry="work", target_identity="synthetic",
    ).request
    node = request.nodes[-1]
    bad_bias_map = replace(node.index_maps[1], coefficients=((1, 0),))
    changed = replace(node, index_maps=(node.index_maps[0], bad_bias_map, node.index_maps[2]))
    with pytest.raises(ValueError, match="not a singleton projection"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], changed)), {
            "arg0": TensorValue(request.node("arg0").type, (0,) * 6),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 3),
        })


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("(i,j)->(j)", "(i,j)->(i)"),
        ("%a, %b : f32", "%a, %old : f32"),
        ("arith.addf %a, %b : f32",
         '"arith.addf"(%a, %b) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32'),
    ],
)
def test_projected_f32_add_rejects_changed_map_body_or_flags(old: str, new: str):
    with pytest.raises(LinalgBridgeError, match="f32 linalg.generic needs an admitted semantic decomposition"):
        translate_linalg_text(
            _f32_projected_binary_source("addf").replace(old, new),
            entry="work", target_identity="synthetic",
        )


def _f32_reduction_source(*, order: str = "") -> str:
    return '''module {
func.func @work(%source: tensor<2x3xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
  %result = "linalg.reduce"(%source, %init) <{dimensions = array<i64: 1>}> ({
    ^bb0(%value: f32, %acc: f32):
      %sum = "arith.addf"(%value, %acc) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%sum) : (f32) -> ()
  }) ORDER : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %result : tensor<2xf32>
}
}'''.replace("ORDER", order)


def test_unqualified_f32_reduction_reports_its_numerical_boundary():
    request = translate_linalg_text(
        _f32_reduction_source(), entry="work", target_identity="synthetic",
    ).request
    node = request.nodes[-1]
    assert node.op == "source_f32_reduce_add_unqualified"
    assert dict(node.attrs) == {
        "axis": 1, "reduction_order": "unresolved",
        "numeric_semantics": "source-f32-add-unqualified-v1",
        "scalar_add_order": "input-accumulator",
    }
    assert node.inputs == ("arg0", "arg1") and request.outputs == (node.id,)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    values = {
        "arg0": TensorValue(request.node("arg0").type, (0,) * 6),
        "arg1": TensorValue(request.node("arg1").type, (0,) * 2),
    }
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, values)
    result = select_and_allocate(request, (), (), bridge=Path("unavailable-native-bridge"))
    assert result.status == "unsupported_semantics"
    assert "qualified numerical-order policy" in result.reason


def test_unqualified_multi_axis_f32_reduction_preserves_axes_without_inventing_order():
    source = _f32_reduction_source().replace(
        "tensor<2x3xf32>", "tensor<2x3x4xf32>",
    ).replace("array<i64: 1>", "array<i64: 1, 2>")
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    node = request.nodes[-1]
    assert node.op == "source_f32_reduce_add_unqualified"
    assert dict(node.attrs) == {
        "axes": "1,2", "reduction_order": "unresolved",
        "numeric_semantics": "source-f32-add-unqualified-v1",
        "scalar_add_order": "input-accumulator",
    }
    assert node.index_maps[0].apply((1, 2, 3)) == (1, 2, 3)
    assert node.index_maps[1].apply((1, 2, 3)) == (1,)
    with pytest.raises(LinalgBridgeError, match="unsupported axis, shape, or type"):
        translate_linalg_text(
            source.replace("array<i64: 1, 2>", "array<i64: 1, 1>"),
            entry="work", target_identity="synthetic",
        )
    with pytest.raises(LinalgBridgeError, match="unsupported axis, shape, or type"):
        translate_linalg_text(
            _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}')
            .replace("tensor<2x3xf32>", "tensor<2x3x4xf32>")
            .replace("array<i64: 1>", "array<i64: 1, 2>"),
            entry="work", target_identity="synthetic",
        )


def test_unqualified_f32_reduction_retains_reversed_add_operands() -> None:
    source = _f32_reduction_source().replace("%value, %acc", "%acc, %value")
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    assert dict(request.nodes[-1].attrs)["scalar_add_order"] == "accumulator-input"
    assert request.digest() != translate_linalg_text(
        _f32_reduction_source(), entry="work", target_identity="synthetic",
    ).request.digest()
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, {
            "arg0": TensorValue(request.node("arg0").type, (0,) * 6),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 2),
        })
    with pytest.raises(LinalgBridgeError, match="changes its declared scalar add order"):
        translate_linalg_text(
            _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}')
            .replace("%value, %acc", "%acc, %value"),
            entry="work", target_identity="synthetic",
        )


def test_unqualified_f32_maximum_reduction_keeps_scalar_order() -> None:
    source = _f32_reduction_source().replace("arith.addf", "arith.maximumf")
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    node = request.nodes[-1]
    assert node.op == "source_f32_reduce_max_unqualified"
    assert dict(node.attrs) == {
        "axis": 1, "reduction_order": "unresolved",
        "numeric_semantics": "source-f32-maximum-unqualified-v1",
        "scalar_operand_order": "input-accumulator",
    }
    reversed_source = source.replace("%value, %acc", "%acc, %value")
    reversed_request = translate_linalg_text(
        reversed_source, entry="work", target_identity="synthetic",
    ).request
    assert dict(reversed_request.nodes[-1].attrs)["scalar_operand_order"] == "accumulator-input"
    assert reversed_request.digest() != request.digest()
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, {
            "arg0": TensorValue(request.node("arg0").type, (0,) * 6),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 2),
        })
    with pytest.raises(LinalgBridgeError, match="changes its declared scalar add order or operation"):
        translate_linalg_text(
            _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}')
            .replace("arith.addf", "arith.maximumf"),
            entry="work", target_identity="synthetic",
        )


def test_ordered_f32_reduction_keeps_initializer_axis_and_step_rounding():
    request = translate_linalg_text(
        _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}'),
        entry="work", target_identity="synthetic",
    ).request
    reduction = request.nodes[-1]
    assert reduction.op == "f32_reduce_add" and reduction.inputs == ("arg0", "arg1")
    assert dict(reduction.attrs) == {
        "axis": 1, "reduction_order": "axis-ascending",
        "numeric_semantics": "ieee754-f32-add-rne-v1",
    }
    assert reduction.index_maps[0].apply((1, 2)) == (1, 2)
    assert reduction.index_maps[1].apply((1, 2)) == (1,)
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, (
            0x4B800000, 0xCB800000, 0x3F800000,
            0x80000000, 0x00000000, 0x80000000,
        )),
        "arg1": TensorValue(request.node("arg1").type, (0x00000000, 0x80000000)),
    })
    # Cancellation before +1 preserves the final 1; other orders can lose it.
    assert result.elements == (0x3F800000, 0x00000000)
    assert reduction.semantic_key() != replace(reduction, attrs=(
        ("axis", 1), ("reduction_order", "axis-descending"),
        ("numeric_semantics", "ieee754-f32-add-rne-v1"),
    )).semantic_key()
    with pytest.raises(ValueError, match="numerical or indexing contract"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], replace(
            reduction, attrs=(("axis", 1), ("reduction_order", "axis-descending"),
                              ("numeric_semantics", "ieee754-f32-add-rne-v1")),
        ),)), {
            "arg0": TensorValue(request.node("arg0").type, (0,) * 6),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 2),
        })


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("axis-ascending", "axis-descending"),
        ("array<i64: 1>", "array<i64: 0>"),
        ("%value, %acc", "%acc, %value"),
        ("arith.addf", "arith.mulf"),
        ("#arith.fastmath<none>", "#arith.fastmath<fast>"),
    ],
)
def test_ordered_f32_reduction_rejects_changed_order_axis_body_or_initializer(old: str, new: str):
    source = _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}').replace(old, new)
    with pytest.raises((LinalgBridgeError, ValueError)):
        translate_linalg_text(source, entry="work", target_identity="synthetic")


def test_ordered_f32_reduction_rejects_empty_initializer():
    source = _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}')
    source = source.replace('  %result = "linalg.reduce"',
                            '  %empty = tensor.empty() : tensor<2xf32>\n  %result = "linalg.reduce"')
    source = source.replace('"linalg.reduce"(%source, %init)', '"linalg.reduce"(%source, %empty)')
    with pytest.raises(LinalgBridgeError, match="defined input and initializer"):
        translate_linalg_text(source, entry="work", target_identity="synthetic")


def test_ordered_f32_reduction_reference_handles_ties_subnormals_and_zeros():
    source = _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}')
    source = source.replace("2x3xf32", "5x3xf32").replace("2xf32", "5xf32")
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    values = (
        0x3F800000, 0x33800000, 0,  # midpoint rounds to even 1.0
        0x3F800001, 0x33800000, 0,  # odd midpoint rounds up
        0x80000000, 0x80000000, 0x80000000,
        1, 1, 1,
        0x7F7FFFFF, 0x7F7FFFFF, 0,
    )
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, values),
        "arg1": TensorValue(request.node("arg1").type, (0, 0, 0x80000000, 0, 0)),
    })
    assert result.elements == (0x3F800000, 0x3F800002, 0x80000000, 3, 0x7F800000)
    with pytest.raises(ValueError, match="NaN result"):
        evaluate_graph(request, {
            "arg0": TensorValue(request.node("arg0").type, (0x7F800000, 0xFF800000, 0, *values[3:])),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 5),
        })


def test_ordered_f32_reduction_reaches_generated_rule(bridge: Path):
    request = translate_linalg_text(
        _f32_reduction_source(order='{merlin.reduction_order = "axis-ascending"}'),
        entry="work", target_identity="synthetic-reduce-v1",
    ).request
    reduction = request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_strict_f32_reduce", "f32_reduce_add", ("external", "external"), "external",
        "f32", "raw-bit-pattern", (1,),
        input_dtypes=("f32", "f32"),
        input_numerical_policies=("raw-bit-pattern", "raw-bit-pattern"),
        input_ranks=(2, 1), index_maps=reduction.index_maps,
        required_attrs=reduction.attrs, shape_contract="relations",
        shape_equalities=(AxisEquality("out", 0, "in0", 0), AxisEquality("out", 0, "in1", 0)),
    )
    selected = select_and_allocate(
        request, (descriptor,), (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0, "arg1": 1}, fixed_outputs=(2,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    assert not descriptor.accepts(replace(reduction, attrs=()), request.nodes[:2])


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("%value, %value", "%value, %old"),
        ("arith.mulf %value, %value : f32", "arith.addf %value, %value : f32"),
        ("linalg.yield %square : f32", "linalg.yield %old : f32"),
        ('"parallel", "parallel", "parallel"', '"parallel", "parallel", "reduction"'),
        ("(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)",
         "(i,j,k)->(i,j,0)>, affine_map<(i,j,k)->(i,j,k)"),
        ("arith.mulf %value, %value : f32",
         '"arith.mulf"(%value, %value) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32'),
    ],
)
def test_static_f32_square_rejects_changed_body_or_indexing(old: str, new: str):
    with pytest.raises(LinalgBridgeError, match="f32 linalg.generic needs an admitted semantic decomposition"):
        translate_linalg_text(
            _f32_square_source().replace(old, new), entry="work", target_identity="synthetic",
        )


def _indexed_copy_source() -> str:
    return '''module {
func.func @work(%source: tensor<1x1x4x4xf32>) -> tensor<2x2x1x1x3x3xf32> {
  %empty = tensor.empty() : tensor<2x2x1x1x3x3xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(h,w,c,b,y,x)->(b,c,y+h,x+w)>,
                     affine_map<(h,w,c,b,y,x)->(h,w,c,b,y,x)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%source : tensor<1x1x4x4xf32>)
    outs(%empty : tensor<2x2x1x1x3x3xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
  } -> tensor<2x2x1x1x3x3xf32>
  func.return %result : tensor<2x2x1x1x3x3xf32>
}
}'''


def test_static_affine_index_copy_preserves_source_bits_and_output_order():
    translated = translate_linalg_text(_indexed_copy_source(), entry="work", target_identity="synthetic")
    request = translated.request
    node = request.nodes[-1]
    assert node.op == "gather" and node.inputs == ("arg0",)
    source = TensorValue(request.node("arg0").type, tuple(range(16)))
    result, = evaluate_graph(request, {"arg0": source})
    expected = []
    for h in range(2):
        for w in range(2):
            for _c in range(1):
                for _b in range(1):
                    for y in range(3):
                        for x in range(3):
                            expected.append((y + h) * 4 + x + w)
    assert result.elements == tuple(expected)
    assert len(result.elements) == 36


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("y+h,x+w", "y+h+1,x+w"),
        ("(h,w,c,b,y,x)>],", "(w,h,c,b,y,x)>],"),
        ("linalg.yield %value : f32", "linalg.yield %old : f32"),
        ('"parallel", "parallel", "parallel", "parallel", "parallel", "parallel"',
         '"parallel", "parallel", "parallel", "parallel", "parallel", "reduction"'),
    ],
)
def test_static_affine_index_copy_rejects_changed_semantics(old, new):
    with pytest.raises(LinalgBridgeError):
        translate_linalg_text(
            _indexed_copy_source().replace(old, new), entry="work", target_identity="synthetic",
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"sizes": "2, 3"}, "sizes differ"),
        ({"offsets": "-1, 1"}, "offset or stride"),
        ({"strides": "1, 3"}, "exceeds destination"),
        ({"destination": "%dest"}, "uninitialized destination"),
    ],
)
def test_static_insert_slice_rejects_invalid_or_uninitialized_forms(mutation, reason):
    arguments = {"destination": "%destination", **mutation}
    with pytest.raises(LinalgBridgeError, match=reason):
        translate_linalg_text(
            _static_insert_source(**arguments), entry="work", target_identity="synthetic",
        )


def _bf16_fp8_cast_source() -> str:
    return """module {
func.func @pack(%x: tensor<2x3xbf16> {merlin.input_domain = "exact-seven-code-domain"})
  -> tensor<2x3xf8E4M3FN> {
  %empty = tensor.empty() : tensor<2x3xf8E4M3FN>
  %result = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<2x3xbf16>) outs(%empty : tensor<2x3xf8E4M3FN>)
    attrs = {merlin.numeric_semantics = "declared-exact-fp8-policy",
             merlin.input_domain = "exact-seven-code-domain",
             merlin.scale_code = 127 : i32} {
    ^bb0(%value: bf16, %old: f8E4M3FN):
      %converted = arith.truncf %value : bf16 to f8E4M3FN
      linalg.yield %converted : f8E4M3FN
    } -> tensor<2x3xf8E4M3FN>
  func.return %result : tensor<2x3xf8E4M3FN>
}
}"""


def test_parsed_bf16_fp8_cast_retains_declared_policy_and_rejects_wrong_body() -> None:
    source = _bf16_fp8_cast_source()
    translated = translate_linalg_text(source, entry="pack", target_identity="synthetic")
    assert translated.source_operations == ("tensor.empty", "linalg.generic", "func.return")
    assert translated.request.outputs == ("op1r0",)
    assert translated.request.nodes[0].attrs == (("input_domain", "exact-seven-code-domain"),)
    converted = translated.request.nodes[1]
    assert (converted.op, converted.type.shape, converted.type.dtype) == (
        "pack_fp8", (2, 3), "fp8_e4m3"
    )
    assert converted.type.numerical_policy == "declared-exact-fp8-policy"
    assert dict(converted.attrs) == {"input_domain": "exact-seven-code-domain", "scale_code": 127}
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    for changed, reason in (
        (source.replace("linalg.yield %converted", "linalg.yield %old"), "does not yield"),
        (source.replace("affine_map<(i,j)->(i,j)>],", "affine_map<(i,j)->(j,i)>],"), "identity maps"),
        (source.replace("merlin.scale_code = 127", "merlin.scale_code = 256"), "scale code"),
        (source.replace("%value : bf16 to", "%old : f8E4M3FN to"), "wrong element types|does not yield|Linalg"),
        (source.replace("exact-seven-code-domain\"})", "another-domain\"})"), "source boundary"),
    ):
        with pytest.raises(LinalgBridgeError, match=reason):
            translate_linalg_text(changed, entry="pack", target_identity="synthetic")


def test_scaled_bf16_fp8_cast_requires_explicit_inverse_scale() -> None:
    unit = _bf16_fp8_cast_source()
    scaled = unit.replace("merlin.scale_code = 127", "merlin.scale_code = 122").replace(
        "%converted = arith.truncf %value : bf16 to f8E4M3FN",
        "%factor = arith.constant 3.200000e+01 : bf16\n"
        "      %scaled = arith.mulf %value, %factor : bf16\n"
        "      %converted = arith.truncf %scaled : bf16 to f8E4M3FN",
    )
    translated = translate_linalg_text(scaled, entry="pack", target_identity="synthetic")
    assert dict(translated.request.nodes[-1].attrs)["scale_code"] == 122
    for changed, reason in (
        (unit.replace("merlin.scale_code = 127", "merlin.scale_code = 122"), "inverse scale"),
        (scaled.replace("3.200000e+01", "1.600000e+01"), "wrong inverse scale"),
        (scaled.replace("arith.mulf %value, %factor", "arith.addf %value, %factor"), "inverse scale"),
        (scaled.replace("arith.truncf %scaled", "arith.truncf %value"), "does not yield"),
    ):
        with pytest.raises(LinalgBridgeError, match=reason):
            translate_linalg_text(changed, entry="pack", target_identity="synthetic")


def test_parsed_relu_pack_keeps_computed_producer_and_declared_premise() -> None:
    source = _bf16_fp8_cast_source().replace(
        "  %empty = tensor.empty() : tensor<2x3xf8E4M3FN>",
        """  %relu_empty = tensor.empty() : tensor<2x3xbf16>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<2x3xbf16>) outs(%relu_empty : tensor<2x3xbf16>)
    attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
             merlin.input_domain = "finite-normal-or-zero-bf16"} {
    ^bb0(%value: bf16, %old: bf16):
      %zero = arith.constant 0.000000e+00 : bf16
      %positive = arith.maximumf %value, %zero : bf16
      linalg.yield %positive : bf16
    } -> tensor<2x3xbf16>
  %empty = tensor.empty() : tensor<2x3xf8E4M3FN>""",
    ).replace(
        "ins(%x : tensor<2x3xbf16>) outs(%empty : tensor<2x3xf8E4M3FN>)",
        "ins(%relu : tensor<2x3xbf16>) outs(%empty : tensor<2x3xf8E4M3FN>)",
    )
    translated = translate_linalg_text(source, entry="pack", target_identity="synthetic")
    assert tuple(node.op for node in translated.request.nodes) == ("input", "relu", "pack_fp8")
    assert translated.request.nodes[-1].inputs == (translated.request.nodes[1].id,)
    assert dict(translated.request.nodes[-1].attrs)["input_domain"] == "exact-seven-code-domain"
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()


def test_parsed_generic_preserves_init_multiple_outputs_and_source_identity() -> None:
    source = _generic_source(two_outputs=True)
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-linalg-v1")
    assert translated.request.source_identity == sha256(source.encode()).hexdigest()
    assert translated.request.outputs == ("op0r0", "arg2")
    assert translated.source_operations == ("linalg.generic", "func.return")
    assert KernelRequest.from_record(translated.request.record()).record() == translated.request.record()
    result = evaluate_graph(translated.request, _inputs())
    assert [value.elements for value in result] == [(32, 38, 59, 74), (10, 10, 10, 10)]
    overflow = evaluate_graph(translated.request, _inputs(init=(1 << 31) - 1))
    assert overflow[0].elements[0] == -(1 << 31) + 21


def test_parsed_fill_materializes_declared_constant_and_rejects_uninitialized_init() -> None:
    translated = translate_linalg_text(_generic_source(init="filled"), entry="work", target_identity="synthetic")
    assert translated.source_operations == (
        "tensor.empty",
        "arith.constant",
        "linalg.fill",
        "linalg.generic",
        "func.return",
    )
    assert len(translated.constants) == 1
    assert translated.constants["op2r0"].elements == (7, 7, 7, 7)
    assert translated.request.constants[0].data_hex == "07000000" * 4
    inputs = {key: value for key, value in _inputs().items() if key != "arg2"}
    assert evaluate_graph(translated.request, inputs, constants=translated.constants)[0].elements == (
        29,
        35,
        56,
        71,
    )
    with pytest.raises(LinalgBridgeError, match="uninitialized"):
        translate_linalg_text(_generic_source(init="empty"), entry="work", target_identity="synthetic")


def test_parsed_tensor_splat_keeps_exact_constant_bits_and_output_order(bridge: Path) -> None:
    source = """module {
func.func @work() -> (tensor<2x3xf32>, tensor<2x3xi32>) {
  %f = arith.constant -0.000000e+00 : f32
  %ft = tensor.splat %f {prov.region_id = "public_splat"} : tensor<2x3xf32>
  %i = arith.constant -7 : i32
  %it = tensor.splat %i : tensor<2x3xi32>
  func.return %ft, %it : tensor<2x3xf32>, tensor<2x3xi32>
}
}"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    assert translated.source_operations == (
        "arith.constant", "tensor.splat", "arith.constant", "tensor.splat", "func.return",
    )
    assert translated.request.outputs == ("op1r0", "op3r0")
    bindings = {binding.node_id: binding for binding in translated.request.constants}
    assert (bindings["op1r0"].encoding, bindings["op1r0"].data_hex) == (
        "f32-le-raw", "00000080" * 6,
    )
    assert (bindings["op3r0"].encoding, bindings["op3r0"].data_hex) == (
        "i32-le", "f9ffffff" * 6,
    )
    assert [value.elements for value in evaluate_graph(translated.request, {}, constants=translated.constants)] == [
        (0x80000000,) * 6, (-7,) * 6,
    ]
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    request = replace(translated.request, physical_extents=(("op1r0", 6), ("op3r0", 6)))
    selected = select_and_allocate(
        request, (), (StorageBank("external", "dram", 12, "word"),),
        bridge=bridge, fixed_outputs=(0, 6),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 0
    assert selected.allocation is not None and set(selected.allocation.addresses.values()) == {0, 6}

    with pytest.raises(LinalgBridgeError, match="unsupported signature or attributes"):
        translate_linalg_text(
            source.replace('{prov.region_id = "public_splat"}', '{unknown = 1 : i32}'),
            entry="work", target_identity="synthetic",
        )


def test_signed_i64_scalar_cast_rounds_once_before_splat() -> None:
    values = (0, 16777217, 16777219, -16777219, -(1 << 63))
    expected = (0x00000000, 0x4B800000, 0x4B800002, 0xCB800002, 0xDF000000)
    body = []
    for index, value in enumerate(values):
        body.extend((
            f"%i{index} = arith.constant {value} : i64",
            f"%f{index} = arith.sitofp %i{index} : i64 to f32",
            f"%t{index} = tensor.splat %f{index} : tensor<2xf32>",
        ))
    results = ", ".join(f"%t{index}" for index in range(len(values)))
    result_types = ", ".join("tensor<2xf32>" for _ in values)
    source = (f"module {{ func.func @work() -> ({result_types}) {{\n"
              + "\n".join(body) + f"\nfunc.return {results} : {result_types}\n}} }}")
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    outputs = evaluate_graph(translated.request, {}, constants=translated.constants)
    assert tuple(result.elements for result in outputs) == tuple((bits, bits) for bits in expected)
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    with pytest.raises(LinalgBridgeError, match="declared constant"):
        translate_linalg_text(
            source.replace("%i1 = arith.constant 16777217 : i64",
                           "%i1 = arith.constant 16777217 : i32")
            .replace("%f1 = arith.sitofp %i1 : i64 to f32",
                     "%f1 = arith.sitofp %i1 : i32 to f32"),
            entry="work", target_identity="synthetic",
        )
    with pytest.raises(LinalgBridgeError, match="declared constant"):
        translate_linalg_text(
            source.replace("%f1 = arith.sitofp %i1 : i64 to f32",
                           "%f1 = arith.sitofp %i1 {unknown = 1 : i32} : i64 to f32"),
            entry="work", target_identity="synthetic",
        )


def test_parsed_bf16_zero_splat_is_initialized_and_rejects_negative_zero() -> None:
    source = """module {
func.func @work() -> tensor<2x3xbf16> {
  %zero = arith.constant 0.000000e+00 : bf16
  %result = tensor.splat %zero : tensor<2x3xbf16>
  func.return %result : tensor<2x3xbf16>
}
}"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    assert translated.request.constants[0].encoding == "bf16-le-raw"
    assert translated.request.constants[0].data_hex == "0000" * 6
    assert evaluate_graph(translated.request, {}, constants=translated.constants)[0].elements == (0,) * 6
    with pytest.raises(LinalgBridgeError, match="constant|admitted"):
        translate_linalg_text(
            source.replace("0.000000e+00", "-0.000000e+00"), entry="work", target_identity="synthetic",
        )


def test_parsed_static_reshape_preserves_bits_and_uses_generated_rules(bridge: Path) -> None:
    source = """module { func.func @work(%a: tensor<6xf32>)
    -> (tensor<2x3xf32>, tensor<6xf32>) {
  %e = tensor.expand_shape %a [[0, 1]] output_shape [2, 3]
    : tensor<6xf32> into tensor<2x3xf32>
  %c = tensor.collapse_shape %e [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  func.return %e, %c : tensor<2x3xf32>, tensor<6xf32>
} }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    assert translated.source_operations == ("tensor.expand_shape", "tensor.collapse_shape", "func.return")
    assert tuple(node.op for node in translated.request.nodes) == ("input", "reshape", "reshape")
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    bits = (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFFFFFFF)
    value = TensorValue(TensorType((6,), "f32", "raw-bit-pattern"), bits)
    assert [result.elements for result in evaluate_graph(translated.request, {"arg0": value})] == [bits, bits]

    def descriptor(name: str, output: tuple[int, ...], operand: tuple[int, ...]) -> InstructionDescriptor:
        return InstructionDescriptor(
            name, "reshape", ("external",), "external", "f32", "raw-bit-pattern", (len(output),),
            input_dtypes=("f32",), input_numerical_policies=("raw-bit-pattern",),
            input_ranks=(len(operand),), shape_contract="bounded",
            output_axis_bounds=tuple(AxisBound(axis, extent, extent) for axis, extent in enumerate(output)),
            input_axis_bounds=(
                tuple(AxisBound(axis, extent, extent) for axis, extent in enumerate(operand)),
            ),
        )

    selected = select_and_allocate(
        translated.request,
        (descriptor("synthetic_expand", (2, 3), (6,)), descriptor("synthetic_collapse", (6,), (2, 3))),
        (StorageBank("external", "dram", 3, "word"),), bridge=bridge,
        fixed_inputs={"arg0": 0}, fixed_outputs=(1, 2),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 2
    assert selected.allocation is not None and set(selected.allocation.addresses.values()) == {0, 1, 2}

    for changed, reason in (
        (source.replace("[[0, 1]] output_shape", "[[1, 0]] output_shape"), "reassociation"),
        (source.replace("tensor.expand_shape %a", "tensor.expand_shape %empty").replace(
            "  %e =", "  %empty = tensor.empty() : tensor<6xf32>\n  %e =", 1,
        ), "uninitialized"),
        (source.replace("output_shape [2, 3]", "output_shape [2, 3] {bad = 1 : i32}"),
         "unsupported signature or attributes"),
    ):
        with pytest.raises(LinalgBridgeError, match=reason):
            translate_linalg_text(changed, entry="work", target_identity="synthetic")


def test_declared_contiguous_reshape_views_select_without_an_instruction(bridge: Path, tmp_path: Path) -> None:
    source = """module { func.func @work(%a: tensor<6xf32>)
    -> (tensor<2x3xf32>, tensor<6xf32>) {
  %e = tensor.expand_shape %a [[0, 1]] output_shape [2, 3]
    : tensor<6xf32> into tensor<2x3xf32>
  %c = tensor.collapse_shape %e [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  func.return %e, %c : tensor<2x3xf32>, tensor<6xf32>
} }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    request = replace(
        translated.request, physical_extents=(("arg0", 6),),
        boundary_layouts=(("arg0", "contiguous_element_order"),),
        output_layouts=("contiguous_element_order", "contiguous_element_order"),
    )
    source_type = request.node("arg0").type
    expanded_type = request.node("op0r0").type
    views = (
        ViewDescriptor("expand", "external", source_type, expanded_type, 6),
        ViewDescriptor("collapse", "external", expanded_type, source_type, 6),
    )
    banks = (StorageBank("external", "dram", 6, "element"),)
    assert tuple(ViewDescriptor.from_record(view.record()) for view in views) == views
    assert len(generate_rules(request, (), views).rewrites) == 2
    unused_copy = InstructionDescriptor(
        "copy", "identity", ("external",), "external", "f32", "raw-bit-pattern", (1,),
        input_dtypes=("f32",), input_numerical_policies=("raw-bit-pattern",),
        value_preserving_copy=True, extent=6,
    )
    profile = NativeTargetProfile("synthetic", (unused_copy,), banks, views)
    assert profile.record()["schema"] == "merlin.native_target_profile.v7"
    assert NativeTargetProfile.from_record(profile.record()) == profile
    with pytest.raises(ValueError, match="declared views"):
        NativeTargetProfile.from_record({**profile.record(), "views": []})
    with pytest.raises(ValueError, match="undeclared storage"):
        NativeTargetProfile("synthetic", (unused_copy,), banks, (replace(views[0], storage="missing"),))
    snapshot = build_native_snapshot(
        profile, destination=tmp_path / "snapshot",
        crate=repo_root() / "src/merlin/semantic_compiler/egg_bridge",
        cargo_target_dir=tmp_path / "cargo", source_revision="public-reshape-view-test",
    )
    assert open_native_snapshot(snapshot.root).select(
        request, fixed_inputs={"arg0": 0}, fixed_outputs=(0, 0),
    ).status == "selected"

    selected = select_and_allocate(
        request, (), banks,
        bridge=bridge, views=views, fixed_inputs={"arg0": 0}, fixed_outputs=(0, 0),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 0
    assert selected.graph is not None and [value.kind for value in selected.graph.values] == ["input", "view", "view"]
    assert selected.allocation is not None and selected.allocation.addresses == {0: 0, 1: 0, 2: 0}
    moved = {**selected.allocation.addresses, 1: 1}
    assert check_assignment(selected.graph, (), moved, (StorageBank("external", "dram", 7, "element"),))[1] == (
        "identity view moved from its source address"
    )
    assert selected.rules is not None and selected.exploration is not None
    assert selected.allocation is not None and selected.candidate is not None
    changed_view = replace(views[0], extent=5)
    changed = check_selection(
        request, (), selected.rules, selected.exploration, selected.candidate,
        selected.graph, selected.allocation, banks, views=(changed_view, views[1]),
        fixed_inputs={"arg0": 0}, fixed_outputs=(0, 0),
    )
    assert not changed.valid and "differs from the target descriptor" in changed.reason
    smaller_input = replace(request, physical_extents=(("arg0", 5),))
    refused = select_and_allocate(
        smaller_input, (), banks, bridge=bridge, views=views,
        fixed_inputs={"arg0": 0}, fixed_outputs=(0, 0),
    )
    assert refused.status != "selected"

    wrong_shape = replace(views[0], result_type=TensorType((3, 2), "f32", "raw-bit-pattern"))
    changed_rules = generate_rules(request, (), (wrong_shape, views[1])).rewrites
    assert not any(rule.name.startswith("view_expand") for rule in changed_rules)
    with pytest.raises(ValueError, match="element bits and count"):
        replace(views[0], result_type=TensorType((2, 4), "f32", "raw-bit-pattern"))
    with pytest.raises(ValueError, match="element bits and count"):
        replace(views[0], result_type=TensorType((2, 3), "f32", "different-rounding"))
    with pytest.raises(ValueError, match="contiguous element order"):
        replace(views[0], physical_layout="unknown")


def test_parsed_copy_preserves_value_and_rejects_implicit_cast(bridge: Path) -> None:
    source = """module {
func.func @work(%a: tensor<2x3xi8>) -> tensor<2x3xi8> {
  %empty = tensor.empty() : tensor<2x3xi8>
  %copy = linalg.copy ins(%a : tensor<2x3xi8>)
    outs(%empty : tensor<2x3xi8>) -> tensor<2x3xi8>
  func.return %copy : tensor<2x3xi8>
}
}"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    assert translated.source_operations == ("tensor.empty", "linalg.copy", "func.return")
    assert translated.request.outputs == ("op1r0",)
    source_value = TensorValue(TensorType((2, 3), "i8", "signed-i8"), (-3, -2, -1, 0, 1, 2))
    assert evaluate_graph(translated.request, {"arg0": source_value}) == (source_value,)
    descriptor = InstructionDescriptor(
        "synthetic_copy", "identity", ("external",), "external", "i8", "signed-i8", (2,),
        input_dtypes=("i8",), input_numerical_policies=("signed-i8",), input_ranks=(2,),
        value_preserving_copy=True,
    )
    selected = select_and_allocate(
        translated.request, (descriptor,), (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0}, fixed_outputs=(1,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    i32_source = source.replace("xi8", "xi32")
    i32_request = translate_linalg_text(i32_source, entry="work", target_identity="synthetic")
    i32_value = TensorValue(TensorType((2, 3), "i32", "i32-wrap-k-ascending"), (1, 2, 3, 4, 5, 6))
    assert evaluate_graph(i32_request.request, {"arg0": i32_value}) == (i32_value,)
    uninitialized = source.replace("ins(%a : tensor<2x3xi8>)", "ins(%empty : tensor<2x3xi8>)")
    with pytest.raises(LinalgBridgeError, match="uninitialized"):
        translate_linalg_text(uninitialized, entry="work", target_identity="synthetic")
    cast_source = source.replace("tensor<2x3xi8>\n  %copy", "tensor<2x3xi32>\n  %copy")
    cast_source = cast_source.replace("outs(%empty : tensor<2x3xi8>) -> tensor<2x3xi8>",
                                      "outs(%empty : tensor<2x3xi32>) -> tensor<2x3xi32>")
    cast_source = cast_source.replace("func.return %copy : tensor<2x3xi8>",
                                      "func.return %copy : tensor<2x3xi32>")
    cast_source = cast_source.replace("-> tensor<2x3xi8> {", "-> tensor<2x3xi32> {")
    with pytest.raises(LinalgBridgeError, match="type-preserving"):
        translate_linalg_text(cast_source, entry="work", target_identity="synthetic")


def test_bf16_copy_and_transpose_preserve_bits_without_admitting_arithmetic(bridge: Path) -> None:
    source = """module { func.func @work(%a: tensor<2x3xbf16>) -> (tensor<2x3xbf16>, tensor<3x2xbf16>) {
      %copy_storage = tensor.empty() : tensor<2x3xbf16>
      %copy = linalg.copy ins(%a : tensor<2x3xbf16>)
        outs(%copy_storage : tensor<2x3xbf16>) -> tensor<2x3xbf16>
      %transposed_storage = tensor.empty() : tensor<3x2xbf16>
      %transposed = linalg.transpose ins(%copy : tensor<2x3xbf16>)
        outs(%transposed_storage : tensor<3x2xbf16>) permutation = [1, 0]
      func.return %copy, %transposed : tensor<2x3xbf16>, tensor<3x2xbf16>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-bf16-movement")
    assert translated.source_operations == (
        "tensor.empty", "linalg.copy", "tensor.empty", "linalg.transpose", "func.return"
    )
    tensor = TensorType((2, 3), "bf16", "raw-bit-pattern")
    bits = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF)
    results = evaluate_graph(translated.request, {"arg0": TensorValue(tensor, bits)})
    assert [result.elements for result in results] == [bits, (bits[0], bits[3], bits[1], bits[4], bits[2], bits[5])]
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()

    transpose = translated.request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_bf16_transpose", "transpose", ("external",), "external", "bf16", "raw-bit-pattern", (2,),
        input_dtypes=("bf16",), input_numerical_policies=("raw-bit-pattern",), input_ranks=(2,),
        index_maps=transpose.index_maps, shape_contract="relations",
        shape_equalities=(AxisEquality("in0", 0, "out", 1), AxisEquality("in0", 1, "out", 0)),
    )
    selected = select_and_allocate(
        translated.request, (descriptor,), (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge, fixed_inputs={"arg0": 0}, fixed_outputs=(None, 2),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1

    arithmetic = source.replace(
        "func.return %copy, %transposed : tensor<2x3xbf16>, tensor<3x2xbf16>",
        "%sum = arith.addf %copy, %copy : tensor<2x3xbf16>\n"
        "      func.return %sum, %transposed : tensor<2x3xbf16>, tensor<3x2xbf16>",
    )
    with pytest.raises(LinalgBridgeError, match="no admitted translation"):
        translate_linalg_text(arithmetic, entry="work", target_identity="synthetic-bf16-movement")


def test_f32_movement_preserves_bits_and_matmul_remains_source_only() -> None:
    source = """module { func.func @work(%a: tensor<2x3xf32>)
      -> (tensor<2x3xf32>, tensor<3x2xf32>, tensor<2x3xf32>) {
      %copy_storage = tensor.empty() : tensor<2x3xf32>
      %copy = linalg.copy ins(%a : tensor<2x3xf32>)
        outs(%copy_storage : tensor<2x3xf32>) -> tensor<2x3xf32>
      %transpose_storage = tensor.empty() : tensor<3x2xf32>
      %transpose = linalg.transpose ins(%copy : tensor<2x3xf32>)
        outs(%transpose_storage : tensor<3x2xf32>) permutation = [1, 0]
      %negative_zero = arith.constant -0.000000e+00 : f32
      %fill_storage = tensor.empty() : tensor<2x3xf32>
      %fill = linalg.fill ins(%negative_zero : f32)
        outs(%fill_storage : tensor<2x3xf32>) -> tensor<2x3xf32>
      func.return %copy, %transpose, %fill
        : tensor<2x3xf32>, tensor<3x2xf32>, tensor<2x3xf32>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-f32-movement")
    assert translated.request.outputs == ("op1r0", "op3r0", "op6r0")
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    assert translated.request.constants[0].encoding == "f32-le-raw"
    assert translated.request.constants[0].data_hex == "00000080" * 6
    bits = (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFFFFFFF)
    value = TensorValue(TensorType((2, 3), "f32", "raw-bit-pattern"), bits)
    result = evaluate_graph(translated.request, {"arg0": value}, constants=translated.constants)
    assert tuple(tensor.elements for tensor in result) == (
        bits,
        (bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]),
        (0x80000000,) * 6,
    )
    with pytest.raises(ValueError, match="outside its admitted domain"):
        TensorValue(value.type, (*bits[:-1], 1 << 32))

    matmul = """module { func.func @work(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>,
      %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %result = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>)
        outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
      func.return %result : tensor<2x2xf32>
    } }"""
    source_graph = translate_linalg_text(matmul, entry="work", target_identity="synthetic-f32-movement")
    node = source_graph.request.nodes[-1]
    assert node.op == "source_f32_matmul_unqualified"
    assert node.inputs == ("arg0", "arg1", "arg2")
    assert dict(node.attrs) == {
        "numeric_semantics": "source-f32-mul-add-unqualified-v1",
        "reduction_order": "unresolved",
    }
    assert node.index_maps and source_graph.request.outputs == (node.id,)
    assert KernelRequest.from_record(source_graph.request.record()).digest() == source_graph.request.digest()
    zero_inputs = {
        f"arg{index}": TensorValue(TensorType(shape, "f32", "raw-bit-pattern"), (0,) * count)
        for index, (shape, count) in enumerate((((2, 3), 6), ((3, 2), 6), ((2, 2), 4)))
    }
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(source_graph.request, zero_inputs)
    unsafe_descriptor = InstructionDescriptor(
        "unsafe_source_matmul", node.op, ("external",) * 3, "external", "f32", "raw-bit-pattern", (2,),
        input_dtypes=("f32",) * 3, input_numerical_policies=("raw-bit-pattern",) * 3,
        input_ranks=(2,) * 3, index_maps=node.index_maps, shape_contract="relations",
        shape_equalities=(
            AxisEquality("in0", 1, "in1", 0), AxisEquality("in0", 0, "out", 0),
            AxisEquality("in1", 1, "out", 1),
        ),
    )
    selection = select_and_allocate(
        source_graph.request, (unsafe_descriptor,), (StorageBank("external", "dram", 64, "word"),),
        bridge=Path("unavailable-native-bridge"), fixed_inputs={"arg0": 0, "arg1": 8, "arg2": 16},
    )
    assert selection.status == "unsupported_semantics"
    assert "explicit qualified numerical transform" in selection.reason


def test_rank_one_concat_uses_checked_adjacent_view(bridge: Path, tmp_path: Path) -> None:
    source = '''module { func.func @work(%a: tensor<2xf32>, %b: tensor<3xf32>)
      -> tensor<5xf32> {
      %joined = "tensor.concat"(%a, %b) <{dim = 0 : i64}>
        : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
      func.return %joined : tensor<5xf32>
    } }'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-concat-view")
    request = replace(
        translated.request, physical_extents=(("arg0", 2), ("arg1", 3)),
        boundary_layouts=(("arg0", "contiguous_element_order"),
                          ("arg1", "contiguous_element_order")),
        output_layouts=("contiguous_element_order",),
    )
    assert request.record()["schema"] == "merlin.semantic_kernel.v4"
    assert KernelRequest.from_record(request.record()) == request
    view = ConcatViewDescriptor(
        "adjacent", "external", (request.nodes[0].type, request.nodes[1].type),
        request.nodes[2].type, (2, 3), 5,
    )
    assert ViewDescriptor.from_record(view.record()) == view
    banks = (StorageBank("external", "dram", 5, "element"),)
    unused = InstructionDescriptor(
        "unrelated", "unrelated", ("external",), "external", "f32", "raw-bit-pattern", (1,),
        input_dtypes=("f32",), input_numerical_policies=("raw-bit-pattern",), extent=1,
    )
    profile = NativeTargetProfile("synthetic-concat-view", (unused,), banks, (view,))
    assert NativeTargetProfile.from_record(profile.record()) == profile
    snapshot = build_native_snapshot(
        profile, destination=tmp_path / "snapshot",
        crate=repo_root() / "src/merlin/semantic_compiler/egg_bridge",
        cargo_target_dir=tmp_path / "cargo", source_revision="public-concat-view-test",
    )
    assert open_native_snapshot(snapshot.root).select(
        request, fixed_inputs={"arg0": 0, "arg1": 2}, fixed_outputs=(0,),
    ).status == "selected"
    selected = select_and_allocate(
        request, (), banks, bridge=bridge, views=(view,),
        fixed_inputs={"arg0": 0, "arg1": 2}, fixed_outputs=(0,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 0
    assert selected.graph is not None and selected.allocation is not None
    assert [value.kind for value in selected.graph.values] == ["input", "input", "view"]
    assert selected.allocation.addresses == {0: 0, 1: 2, 2: 0}
    assert selected.rules is not None and selected.exploration is not None
    changed_layout = replace(request, boundary_layouts=(
        ("arg0", "contiguous_element_order"), ("arg1", "interleaved"),
    ))
    replay = check_selection(
        changed_layout, (), selected.rules, selected.exploration,
        selected.candidate, selected.graph, selected.allocation, banks, views=(view,),
        fixed_inputs={"arg0": 0, "arg1": 2}, fixed_outputs=(0,),
    )
    assert not replay.valid and "physical layout" in replay.reason
    for changed in (
        replace(request, boundary_layouts=(), output_layouts=()),
        replace(request, boundary_layouts=(("arg0", "contiguous_element_order"),
                                           ("arg1", "interleaved"))),
        replace(request, output_layouts=("interleaved",)),
    ):
        refused = select_and_allocate(
            changed, (), banks, bridge=bridge, views=(view,),
            fixed_inputs={"arg0": 0, "arg1": 2}, fixed_outputs=(0,),
        )
        assert refused.status != "selected"
    with pytest.raises(ValueError, match="declared together"):
        replace(request, output_layouts=())
    assert check_assignment(selected.graph, (), {0: 0, 1: 1, 2: 0}, banks)[1] == (
        "instruction address map or validity failed"
    )
    assert evaluate_graph(request, {
        "arg0": TensorValue(request.nodes[0].type, (0x80000000, 0x7FC00001)),
        "arg1": TensorValue(request.nodes[1].type, (1, 2, 3)),
    })[0].elements == (0x80000000, 0x7FC00001, 1, 2, 3)
    moved = select_and_allocate(
        request, (), banks, bridge=bridge, views=(view,),
        fixed_inputs={"arg0": 0, "arg1": 1}, fixed_outputs=(0,),
    )
    assert moved.status == "compile_error"
    reversed_view = replace(
        view, source_types=(view.source_types[1], view.source_types[0]), source_extents=(3, 2),
    )
    assert not generate_rules(request, (), (reversed_view,)).rewrites
    byte_request = replace(request, physical_extents=(("arg0", 8), ("arg1", 12)))
    byte_view = replace(view, source_extents=(8, 12), extent=20)
    byte_banks = (StorageBank("external", "dram", 20, "byte", alignment=4),)
    byte_selected = select_and_allocate(
        byte_request, (), byte_banks, bridge=bridge, views=(byte_view,),
        fixed_inputs={"arg0": 0, "arg1": 8}, fixed_outputs=(0,),
    )
    assert byte_selected.status == "selected", byte_selected.reason
    assert byte_selected.allocation is not None
    assert byte_selected.allocation.addresses == {0: 0, 1: 8, 2: 0}


def test_outer_axis_matrix_concat_uses_adjacent_row_major_spans(bridge: Path) -> None:
    source = '''module { func.func @work(%a: tensor<1x2xf32>, %b: tensor<2x2xf32>)
      -> tensor<3x2xf32> {
      %joined = "tensor.concat"(%a, %b) <{dim = 0 : i64}>
        : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
      func.return %joined : tensor<3x2xf32>
    } }'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-matrix-concat")
    request = replace(
        translated.request, physical_extents=(("arg0", 8), ("arg1", 16)),
        boundary_layouts=(("arg0", "contiguous_element_order"),
                          ("arg1", "contiguous_element_order")),
        output_layouts=("contiguous_element_order",),
    )
    view = ConcatViewDescriptor(
        "row_blocks", "external", (request.nodes[0].type, request.nodes[1].type),
        request.nodes[2].type, (8, 16), 24,
    )
    banks = (StorageBank("external", "dram", 24, "byte", alignment=4),)
    selected = select_and_allocate(
        request, (), banks, bridge=bridge, views=(view,),
        fixed_inputs={"arg0": 0, "arg1": 8}, fixed_outputs=(0,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.allocation is not None
    assert selected.allocation.addresses == {0: 0, 1: 8, 2: 0}
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, (1, 2)),
        "arg1": TensorValue(request.node("arg1").type, (3, 4, 5, 6)),
    })
    assert result.elements == (1, 2, 3, 4, 5, 6)
    with pytest.raises(ValueError, match="element order"):
        ConcatViewDescriptor(
            "interleaved", "external", (TensorType((2, 1), "f32", "raw-bit-pattern"),
                                         TensorType((2, 2), "f32", "raw-bit-pattern")),
            TensorType((2, 3), "f32", "raw-bit-pattern"), (8, 16), 24,
        )


def test_parsed_concat_preserves_order_and_raw_bits() -> None:
    source = '''module { func.func @work(
      %a: tensor<1x2x2xf32>, %b: tensor<1x1x2xf32>, %c: tensor<1x2x2xf32>)
      -> (tensor<1x5x2xf32>, tensor<1x1x2xf32>) {
      %joined = "tensor.concat"(%a, %b, %c) <{dim = 1 : i64}>
        {prov.region_id = "public_concat"}
        : (tensor<1x2x2xf32>, tensor<1x1x2xf32>, tensor<1x2x2xf32>)
          -> tensor<1x5x2xf32>
      func.return %joined, %b : tensor<1x5x2xf32>, tensor<1x1x2xf32>
    } }'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-concat")
    request = translated.request
    assert translated.source_operations == ("tensor.concat", "func.return")
    assert request.outputs == ("op0r0", "arg1")
    assert request.nodes[-1].op == "concat"
    assert request.nodes[-1].inputs == ("arg0", "arg1", "arg2")
    assert request.nodes[-1].attrs == (("axis", 1),)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    tensors = {
        "arg0": TensorValue(request.node("arg0").type, (0x80000000, 0x7FC00001, 1, 2)),
        "arg1": TensorValue(request.node("arg1").type, (0x7F800000, 0xFF800000)),
        "arg2": TensorValue(request.node("arg2").type, (3, 4, 5, 6)),
    }
    result = evaluate_graph(request, tensors)
    assert result[0].elements == (*tensors["arg0"].elements, *tensors["arg1"].elements,
                                  *tensors["arg2"].elements)
    assert result[1].elements == tensors["arg1"].elements
    for changed in (
        source.replace("dim = 1 : i64", "dim = 3 : i64"),
        source.replace("tensor<1x5x2xf32>", "tensor<1x6x2xf32>"),
        source.replace('{prov.region_id = "public_concat"}', '{unqualified = 1 : i64}'),
    ):
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-concat")
    with pytest.raises(ValueError, match="reference concat"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], replace(
            request.nodes[-1], attrs=(("axis", 0),),
        ))), tensors)


def test_parsed_concat_interleaves_outer_slices() -> None:
    source = '''module { func.func @work(%a: tensor<2x1x2xf32>, %b: tensor<2x2x2xf32>)
      -> tensor<2x3x2xf32> {
      %joined = "tensor.concat"(%a, %b) <{dim = 1 : i64}>
        : (tensor<2x1x2xf32>, tensor<2x2x2xf32>) -> tensor<2x3x2xf32>
      func.return %joined : tensor<2x3x2xf32>
    } }'''
    request = translate_linalg_text(source, entry="work", target_identity="synthetic-concat").request
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, (1, 2, 3, 4)),
        "arg1": TensorValue(request.node("arg1").type, (10, 11, 12, 13, 20, 21, 22, 23)),
    })
    assert result.elements == (1, 2, 10, 11, 12, 13, 3, 4, 20, 21, 22, 23)


def test_batched_f32_contraction_keeps_maps_init_and_scalar_add_order() -> None:
    source = '''module { func.func @work(%a: tensor<2x3x4xf32>,
      %b: tensor<2x4x5xf32>, %init: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
      %result = linalg.generic {
        indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,
                         affine_map<(b,m,n,k)->(b,k,n)>,
                         affine_map<(b,m,n,k)->(b,m,n)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%a, %b : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
        outs(%init : tensor<2x3x5xf32>) {
        ^bb0(%av: f32, %bv: f32, %acc: f32):
          %product = arith.mulf %av, %bv : f32
          %sum = arith.addf %acc, %product : f32
          linalg.yield %sum : f32
      } -> tensor<2x3x5xf32>
      func.return %result : tensor<2x3x5xf32>
    } }'''
    request = translate_linalg_text(source, entry="work", target_identity="synthetic-batched-f32").request
    node = request.nodes[-1]
    assert node.op == "source_f32_matmul_unqualified"
    assert node.inputs == ("arg0", "arg1", "arg2")
    assert dict(node.attrs) == {
        "numeric_semantics": "source-f32-mul-add-unqualified-v1",
        "reduction_order": "unresolved", "scalar_add_order": "accumulator-product",
    }
    assert tuple(mapping.apply((1, 2, 3, 0)) for mapping in node.index_maps) == (
        (1, 2, 0), (1, 0, 3), (1, 2, 3), (1, 2, 3),
    )
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    other_order = source.replace("%acc, %product", "%product, %acc")
    other = translate_linalg_text(other_order, entry="work", target_identity="synthetic-batched-f32").request
    assert dict(other.nodes[-1].attrs)["scalar_add_order"] == "product-accumulator"
    assert other.digest() != request.digest()
    for changed in (
        source.replace("(b,k,n)", "(b,n,k)"),
        source.replace("arith.mulf %av, %bv", "arith.mulf %bv, %av"),
        source.replace("arith.addf %acc, %product", "arith.subf %acc, %product"),
    ):
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-batched-f32")
    with pytest.raises(ValueError, match="unresolved reduction order"):
        evaluate_graph(request, {
            "arg0": TensorValue(request.node("arg0").type, (0,) * 24),
            "arg1": TensorValue(request.node("arg1").type, (0,) * 40),
            "arg2": TensorValue(request.node("arg2").type, (0,) * 30),
        })


def test_parsed_sigmoid_region_remains_source_only_and_checks_scalar_order() -> None:
    source = '''module { func.func @work(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %empty = tensor.empty() : tensor<2x3xf32>
      %result = linalg.generic {
        indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%x : tensor<2x3xf32>) outs(%empty : tensor<2x3xf32>) {
        ^bb0(%value: f32, %old: f32):
          %one = arith.constant 1.000000e+00 : f32
          %negative = arith.negf %value : f32
          %exponential = math.exp %negative : f32
          %denominator = arith.addf %one, %exponential : f32
          %sigmoid = arith.divf %one, %denominator : f32
          linalg.yield %sigmoid : f32
      } -> tensor<2x3xf32>
      func.return %result : tensor<2x3xf32>
    } }'''
    request = translate_linalg_text(source, entry="work", target_identity="synthetic").request
    node = request.nodes[-1]
    assert node.op == "source_f32_composite_unqualified"
    assert node.inputs == ("arg0",)
    assert dict(node.attrs) == {
        "numeric_semantics": "source-f32-composite-unqualified-v1",
        "operation": "sigmoid_f32", "literal_bits": "0x3f800000",
    }
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    with pytest.raises(ValueError, match="no qualified numerical reference"):
        evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, (0,) * 6)})
    selection = select_and_allocate(
        request, (), (StorageBank("external", "dram", 64, "word"),),
        bridge=Path("unavailable-native-bridge"), fixed_inputs={"arg0": 0},
    )
    assert selection.status == "unsupported_semantics"
    assert "qualified numerical transform or compiled host route" in selection.reason
    for changed in (
        source.replace("arith.addf %one, %exponential", "arith.addf %exponential, %one"),
        source.replace("arith.divf %one, %denominator", "arith.divf %denominator, %one"),
        source.replace("1.000000e+00", "2.000000e+00"),
        source.replace("math.exp", "math.erf"),
    ):
        with pytest.raises(LinalgBridgeError, match="admitted semantic decomposition"):
            translate_linalg_text(changed, entry="work", target_identity="synthetic")


def test_rank_four_transpose_preserves_bit_patterns_and_checked_maps() -> None:
    source = '''module { func.func @work(%x: tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32> {
      %empty = tensor.empty() : tensor<1x3x2x4xf32>
      %result = linalg.transpose ins(%x : tensor<1x2x3x4xf32>)
        outs(%empty : tensor<1x3x2x4xf32>) permutation = [0, 2, 1, 3]
      func.return %result : tensor<1x3x2x4xf32>
    } }'''
    request = translate_linalg_text(source, entry="work", target_identity="synthetic-transpose").request
    node = request.nodes[-1]
    assert node.op == "transpose"
    assert node.index_maps[0].apply((0, 2, 1, 3)) == (0, 1, 2, 3)
    assert node.index_maps[1].apply((0, 2, 1, 3)) == (0, 2, 1, 3)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    input_bits = tuple(0x80000000 | index for index in range(24))
    result, = evaluate_graph(request, {
        "arg0": TensorValue(request.node("arg0").type, input_bits),
    })
    assert result.elements == tuple(
        input_bits[((row * 3 + column) * 4 + lane)]
        for column in range(3) for row in range(2) for lane in range(4)
    )
    changed_map = replace(node.index_maps[0], coefficients=((1, 0, 0, 0),) * 4)
    with pytest.raises(ValueError, match="unsupported maps"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], replace(node, index_maps=(
            changed_map, node.index_maps[1],
        )))), {"arg0": TensorValue(request.node("arg0").type, input_bits)})
    for changed in (
        source.replace("[0, 2, 1, 3]", "[0, 2, 2, 3]"),
        source.replace("tensor<1x3x2x4xf32>", "tensor<1x2x3x4xf32>"),
    ):
        with pytest.raises((LinalgBridgeError, ValueError)):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-transpose")


def test_f32_ordered_zero_select_preserves_negative_zero_and_nan_payload() -> None:
    source = """module { func.func @work(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %empty = tensor.empty() : tensor<2x3xf32>
      %result = linalg.generic {
        indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%x : tensor<2x3xf32>) outs(%empty : tensor<2x3xf32>) {
      ^bb0(%value: f32, %old: f32):
        %integer = arith.constant 0 : i64
        %zero = arith.sitofp %integer : i64 to f32
        %negative = arith.cmpf olt, %value, %zero : f32
        %chosen = arith.select %negative, %zero, %value : f32
        linalg.yield %chosen : f32
      } -> tensor<2x3xf32>
      func.return %result : tensor<2x3xf32>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-f32-zero-select")
    node = translated.request.nodes[-1]
    assert node.op == "f32_relu_ordered_select" and node.inputs == ("arg0",)
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    tensor = TensorType((2, 3), "f32", "raw-bit-pattern")
    bits = (0xBF800000, 0x80000000, 0x00000000, 0x3F800000, 0xFF800000, 0xFFC00001)
    result = evaluate_graph(translated.request, {"arg0": TensorValue(tensor, bits)})
    assert result[0].elements == (0, 0x80000000, 0, 0x3F800000, 0, 0xFFC00001)
    with pytest.raises(ValueError, match="signaling NaN"):
        evaluate_graph(translated.request, {
            "arg0": TensorValue(tensor, (*bits[:-1], 0xFF800001)),
        })
    for changed in (
        source.replace("arith.constant 0 : i64", "arith.constant 1 : i64"),
        source.replace("arith.cmpf olt", "arith.cmpf oge"),
        source.replace("%negative, %zero, %value", "%negative, %value, %zero"),
    ):
        with pytest.raises(LinalgBridgeError, match="f32 linalg.generic needs an admitted semantic decomposition"):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-f32-zero-select")


@pytest.mark.parametrize("kind", ("exp", "rsqrt"))
def test_f32_unary_math_is_preserved_but_cannot_be_selected(kind: str) -> None:
    source = f'''module {{ func.func @work(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {{
      %empty = tensor.empty() : tensor<2x3xf32>
      %result = linalg.generic {{
        indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
        iterator_types = ["parallel", "parallel"]
      }} ins(%x : tensor<2x3xf32>) outs(%empty : tensor<2x3xf32>) {{
      ^bb0(%value: f32, %old: f32):
        %computed = math.{kind} %value : f32
        linalg.yield %computed : f32
      }} -> tensor<2x3xf32>
      func.return %result : tensor<2x3xf32>
    }} }}'''
    request = translate_linalg_text(source, entry="work", target_identity="synthetic-math-source").request
    node = request.nodes[-1]
    assert node.op == f"source_f32_{kind}_unqualified"
    assert node.inputs == ("arg0",)
    assert dict(node.attrs) == {"numeric_semantics": f"source-f32-{kind}-unqualified-v1"}
    assert node.index_maps == pointwise_identity_maps(2, 1)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    inputs = {"arg0": TensorValue(request.node("arg0").type, (0,) * 6)}
    with pytest.raises(ValueError, match="no qualified numerical reference"):
        evaluate_graph(request, inputs)
    selected = select_and_allocate(request, (), (), bridge=Path("unavailable-native-bridge"))
    assert selected.status == "unsupported_semantics"
    assert "qualified numerical transform" in selected.reason


def test_parsed_bf16_relu_preserves_maps_and_requires_exact_maximum_body() -> None:
    source = """module { func.func @work(%x: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
      %empty = tensor.empty() : tensor<2x2xbf16>
      %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%x : tensor<2x2xbf16>) outs(%empty : tensor<2x2xbf16>)
        attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
                 merlin.input_domain = "finite-normal-or-zero-bf16"} {
        ^bb0(%value: bf16, %old: bf16):
          %zero = arith.constant 0.000000e+00 : bf16
          %positive = arith.maximumf %value, %zero : bf16
          linalg.yield %positive : bf16
      } -> tensor<2x2xbf16>
      func.return %result : tensor<2x2xbf16>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-bf16-relu",
                                       lowering_policy="diagnostic")
    assert translated.source_operations == ("tensor.empty", "linalg.generic", "func.return")
    assert translated.request.source_identity == sha256(source.encode()).hexdigest()
    assert translated.request.outputs == ("op1r0",)
    assert translated.request.nodes[-1].op == "relu"
    assert len(translated.request.nodes[-1].index_maps) == 2
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()
    bits = (0x8000, 0x0000, 0xBF80, 0x3F80)
    result = evaluate_graph(translated.request, {
        "arg0": TensorValue(TensorType((2, 2), "bf16", "raw-bit-pattern"), bits),
    })
    assert result[0].elements == (0, 0, 0, 0x3F80)

    mutations = (
        ("merlin.numeric_semantics = \"bf16-sign-clamp-v1\"", "merlin.numeric_semantics = \"raw-bit-pattern\""),
        ("merlin.input_domain = \"finite-normal-or-zero-bf16\"", "merlin.input_domain = \"all-bit-patterns\""),
        ("affine_map<(d0, d1)->(d0, d1)>", "affine_map<(d0, d1)->(d1, d0)>"),
        ("arith.maximumf %value, %zero", "arith.minimumf %value, %zero"),
        ("arith.maximumf %value, %zero", "arith.maximumf %old, %zero"),
        ("arith.maximumf %value, %zero : bf16",
         "arith.maximumf %value, %zero fastmath<nsz> : bf16"),
        ("arith.constant 0.000000e+00", "arith.constant -0.000000e+00"),
    )
    for before, after in mutations:
        changed = source.replace(before, after, 1)
        assert changed != source
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-bf16-relu",
                                  lowering_policy="diagnostic")


def _bf16_add_source() -> str:
    return """module { func.func @work(
      %a: tensor<2x2xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"},
      %b: tensor<2x2xbf16> {merlin.input_domain = "finite-normal-exact-fp32-sum-v1"})
      -> tensor<2x2xbf16> {
      %empty = tensor.empty() : tensor<2x2xbf16>
      %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%a, %b : tensor<2x2xbf16>, tensor<2x2xbf16>)
        outs(%empty : tensor<2x2xbf16>)
        attrs = {merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1",
                 merlin.input_domain = "finite-normal-exact-fp32-sum-v1"} {
        ^bb0(%left: bf16, %right: bf16, %old: bf16):
          %lf = arith.extf %left : bf16 to f32
          %rf = arith.extf %right : bf16 to f32
          %sum = arith.addf %lf, %rf : f32
          %bits = arith.bitcast %sum : f32 to i32
          %shift = arith.constant 16 : i32
          %hi = arith.shrui %bits, %shift : i32
          %narrow = arith.trunci %hi : i32 to i16
          %chopped = arith.bitcast %narrow : i16 to bf16
          linalg.yield %chopped : bf16
      } -> tensor<2x2xbf16>
      func.return %result : tensor<2x2xbf16>
    } }"""


def test_parsed_bf16_add_preserves_exact_chop_program_and_pairwise_premise() -> None:
    source = _bf16_add_source()
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-bf16-add",
                                       lowering_policy="diagnostic")
    request = translated.request
    assert translated.source_operations == ("tensor.empty", "linalg.generic", "func.return")
    assert request.outputs == ("op1r0",)
    assert request.nodes[-1].op == "add_bf16"
    assert request.nodes[-1].inputs == ("arg0", "arg1")
    assert dict(request.nodes[-1].attrs) == {
        "input_domain": "finite-normal-exact-fp32-sum-v1",
        "numeric_semantics": "selected-f32-rne-bf16-chop-v1",
    }
    assert len(request.nodes[-1].index_maps) == 3
    assert KernelRequest.from_record(request.record()).digest() == request.digest()

    mutations = (
        ('merlin.numeric_semantics = "selected-f32-rne-bf16-chop-v1"',
         'merlin.numeric_semantics = "bf16-rne-v1"'),
        ('merlin.input_domain = "finite-normal-exact-fp32-sum-v1"',
         'merlin.input_domain = "all-bit-patterns"'),
        ('%rf = arith.extf %right', '%rf = arith.extf %left'),
        ('%sum = arith.addf %lf, %rf : f32',
         '%sum = arith.addf %lf, %rf fastmath<reassoc> : f32'),
        ('%hi = arith.shrui %bits, %shift', '%hi = arith.shrsi %bits, %shift'),
        ('%shift = arith.constant 16', '%shift = arith.constant 15'),
        ('%narrow = arith.trunci %hi', '%narrow = arith.trunci %bits'),
        ('linalg.yield %chopped', 'linalg.yield %old'),
        ('affine_map<(d0, d1)->(d0, d1)>', 'affine_map<(d0, d1)->(d1, d0)>'),
    )
    for before, after in mutations:
        changed = source.replace(before, after, 1)
        assert changed != source
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-bf16-add",
                                  lowering_policy="diagnostic")


def test_parsed_bf16_sub_preserves_operand_order_and_zero_domain() -> None:
    source = (_bf16_add_source()
              .replace("finite-normal-exact-fp32-sum-v1",
                       "finite-normal-exact-fp32-difference-or-zero-v1")
              .replace("%sum = arith.addf", "%sum = arith.subf"))
    request = translate_linalg_text(
        source, entry="work", target_identity="synthetic-bf16-sub",
        lowering_policy="diagnostic",
    ).request
    assert request.nodes[-1].op == "sub_bf16"
    assert request.nodes[-1].inputs == ("arg0", "arg1")
    assert dict(request.nodes[-1].attrs) == {
        "input_domain": "finite-normal-exact-fp32-difference-or-zero-v1",
        "numeric_semantics": "selected-f32-rne-bf16-chop-v1",
    }
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    mutations = (
        ("%sum = arith.subf %lf, %rf", "%sum = arith.subf %rf, %lf"),
        ("%sum = arith.subf %lf, %rf", "%sum = arith.addf %lf, %rf"),
        ("%sum = arith.subf %lf, %rf", "%sum = arith.subf %lf, %rf fastmath<reassoc>"),
        ("finite-normal-exact-fp32-difference-or-zero-v1", "all-bit-patterns"),
    )
    for before, after in mutations:
        changed = source.replace(before, after, 1)
        assert changed != source
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-bf16-sub",
                                  lowering_policy="diagnostic")


def _bf16_mul_source() -> str:
    source = _bf16_add_source().replace(
        "finite-normal-exact-fp32-sum-v1", "finite-normal-bf16-product-normal-v1"
    ).replace("selected-f32-rne-bf16-chop-v1", "bf16-product-rne-v1")
    old = """%sum = arith.addf %lf, %rf : f32
          %bits = arith.bitcast %sum : f32 to i32
          %shift = arith.constant 16 : i32
          %hi = arith.shrui %bits, %shift : i32
          %narrow = arith.trunci %hi : i32 to i16
          %chopped = arith.bitcast %narrow : i16 to bf16
          linalg.yield %chopped : bf16"""
    new = """%product = arith.mulf %lf, %rf : f32
          %bits = arith.bitcast %product : f32 to i32
          %shift = arith.constant 16 : i32
          %half = arith.constant 32767 : i32
          %one = arith.constant 1 : i32
          %high = arith.shrui %bits, %shift : i32
          %lsb = arith.andi %high, %one : i32
          %bias = arith.addi %half, %lsb : i32
          %biased = arith.addi %bits, %bias : i32
          %rounded = arith.shrui %biased, %shift : i32
          %narrow = arith.trunci %rounded : i32 to i16
          %rounded_bf16 = arith.bitcast %narrow : i16 to bf16
          linalg.yield %rounded_bf16 : bf16"""
    assert old in source
    return source.replace(old, new)


def test_parsed_bf16_mul_checks_nearest_even_body_and_normal_domain() -> None:
    source = _bf16_mul_source()
    translated = translate_linalg_text(
        source, entry="work", target_identity="synthetic-bf16-mul", lowering_policy="diagnostic",
    )
    request = translated.request
    assert translated.source_operations == ("tensor.empty", "linalg.generic", "func.return")
    assert request.outputs == ("op1r0",)
    assert tuple(node.op for node in request.nodes) == ("input", "input", "mul_bf16")
    assert request.nodes[-1].inputs == ("arg0", "arg1")
    assert dict(request.nodes[-1].attrs) == {
        "input_domain": "finite-normal-bf16-product-normal-v1",
        "numeric_semantics": "bf16-product-rne-v1",
    }
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    mutations = (
        ('merlin.numeric_semantics = "bf16-product-rne-v1"',
         'merlin.numeric_semantics = "bf16-chop-v1"'),
        ('merlin.input_domain = "finite-normal-bf16-product-normal-v1"',
         'merlin.input_domain = "all-bit-patterns"'),
        ("%rf = arith.extf %right", "%rf = arith.extf %left"),
        ("%product = arith.mulf %lf, %rf : f32",
         "%product = arith.mulf %lf, %rf fastmath<reassoc> : f32"),
        ("%half = arith.constant 32767", "%half = arith.constant 32766"),
        ("%one = arith.constant 1", "%one = arith.constant 2"),
        ("%bias = arith.addi %half, %lsb", "%bias = arith.addi %half, %one"),
        ("%rounded = arith.shrui %biased", "%rounded = arith.shrui %bits"),
        ("linalg.yield %rounded_bf16", "linalg.yield %old"),
        ("affine_map<(d0, d1)->(d0, d1)>", "affine_map<(d0, d1)->(d1, d0)>"),
    )
    for before, after in mutations:
        changed = source.replace(before, after, 1)
        assert changed != source
        with pytest.raises(LinalgBridgeError):
            translate_linalg_text(
                changed, entry="work", target_identity="synthetic-bf16-mul", lowering_policy="diagnostic",
            )


def test_parsed_bf16_add_relu_preserves_shared_value_and_output_order() -> None:
    source = _bf16_add_source().replace(
        "-> tensor<2x2xbf16> {", "-> (tensor<2x2xbf16>, tensor<2x2xbf16>) {", 1,
    ).replace(
        "func.return %result : tensor<2x2xbf16>",
        """%relu_empty = tensor.empty() : tensor<2x2xbf16>
      %relu = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%result : tensor<2x2xbf16>) outs(%relu_empty : tensor<2x2xbf16>)
        attrs = {merlin.numeric_semantics = "bf16-sign-clamp-v1",
                 merlin.input_domain = "finite-normal-or-zero-bf16"} {
        ^bb0(%value: bf16, %old: bf16):
          %zero = arith.constant 0.000000e+00 : bf16
          %positive = arith.maximumf %value, %zero : bf16
          linalg.yield %positive : bf16
      } -> tensor<2x2xbf16>
      func.return %result, %relu : tensor<2x2xbf16>, tensor<2x2xbf16>""",
    )
    request = translate_linalg_text(
        source, entry="work", target_identity="synthetic-bf16-composition",
        lowering_policy="diagnostic",
    ).request
    assert tuple(node.op for node in request.nodes) == ("input", "input", "add_bf16", "relu")
    assert request.nodes[-1].inputs == (request.nodes[-2].id,)
    assert request.outputs == (request.nodes[-2].id, request.nodes[-1].id)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()

    tensor = TensorType((2, 2), "bf16", "raw-bit-pattern")
    left = (0x3F80, 0xBF80, 0x3F80, 0xBF80)
    right = (0x3BC0, 0xBF00, 0x4000, 0xBF00)
    calls = []

    def add(node: SemanticNode, operands: tuple[TensorValue, ...]) -> TensorValue:
        calls.append(node.id)
        assert (operands[0].elements, operands[1].elements) == (left, right)
        return TensorValue(node.type, (0x3F80, 0xBFC0, 0x4040, 0xBFC0))

    inputs = {"arg0": TensorValue(tensor, left), "arg1": TensorValue(tensor, right)}
    primitives = {("add_bf16", "raw-bit-pattern"): add}
    outputs = evaluate_graph(request, inputs, primitives=primitives)
    assert tuple(output.elements for output in outputs) == (
        (0x3F80, 0xBFC0, 0x4040, 0xBFC0),
        (0x3F80, 0, 0x4040, 0),
    )
    assert calls == [request.nodes[-2].id]

    swapped = source.replace(
        "func.return %result, %relu : tensor<2x2xbf16>, tensor<2x2xbf16>",
        "func.return %relu, %result : tensor<2x2xbf16>, tensor<2x2xbf16>",
    )
    swapped_request = translate_linalg_text(
        swapped, entry="work", target_identity="synthetic-bf16-composition",
        lowering_policy="diagnostic",
    ).request
    assert swapped_request.outputs == request.outputs[::-1]
    assert evaluate_graph(swapped_request, inputs, primitives=primitives) == outputs[::-1]


def test_native_snapshot_binds_frontend_and_source_preparation_bytes() -> None:
    from merlin.semantic_compiler import snapshot as snapshot_module

    sources = snapshot_module._compiler_sources()
    assert sources["frontends/linalg_bf16_add.py"] == sha256(
        (repo_root() / "src/merlin/frontends/linalg_bf16_add.py").read_bytes()
    ).hexdigest()
    changed = dict(sources)
    changed["frontends/linalg_bf16_add.py"] = "0" * 64
    with pytest.raises(ValueError, match="differ from snapshot"):
        snapshot_module._check_compiler_sources({"compiler_sources": changed})
    for name in (
        "precision_codegen.py",
        "precision_constants.py",
        "precision_intake.py",
        "precision_transform.py",
        "precision_values.py",
        "frontends/compile_inputs.py",
        "capture/safetensors.py",
        "runtime/fp8_formats.py",
        "xdsl_dialects/_common.py",
        "xdsl_dialects/fp8.py",
    ):
        source = repo_root() / "src/merlin" / (
            name if "/" in name else f"semantic_compiler/{name}"
        )
        assert sources[name] == sha256(source.read_bytes()).hexdigest()
        changed = dict(sources)
        changed[name] = "0" * 64
        with pytest.raises(ValueError, match="differ from snapshot"):
            snapshot_module._check_compiler_sources({"compiler_sources": changed})
    for name in ("ordered_k_tiling.py", "targetgen/cli.py"):
        source = repo_root() / "src/merlin" / (
            "semantic_compiler/ordered_k_tiling.py" if name == "ordered_k_tiling.py" else name
        )
        assert sources[name] == sha256(source.read_bytes()).hexdigest()
        changed = dict(sources)
        changed[name] = "0" * 64
        with pytest.raises(ValueError, match="differ from snapshot"):
            snapshot_module._check_compiler_sources({"compiler_sources": changed})
    assert sources["frontends/linalg_bf16_mul.py"] == sha256(
        (repo_root() / "src/merlin/frontends/linalg_bf16_mul.py").read_bytes()
    ).hexdigest()
    changed = dict(sources)
    changed["frontends/linalg_bf16_mul.py"] = "0" * 64
    with pytest.raises(ValueError, match="differ from snapshot"):
        snapshot_module._check_compiler_sources({"compiler_sources": changed})


def _ordered_fp8_source() -> str:
    return """module { func.func @work(
      %a: tensor<2x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},
      %w: tensor<3x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})
      -> tensor<2x3xbf16> {
      %zero = arith.constant 0.000000e+00 : bf16
      %empty = tensor.empty() : tensor<2x3xbf16>
      %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<2x3xbf16>) -> tensor<2x3xbf16>
      %result = linalg.generic {
        indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                         affine_map<(m,n,k)->(n,k)>,
                         affine_map<(m,n,k)->(m,n)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%a, %w : tensor<2x4xf8E4M3FN>, tensor<3x4xf8E4M3FN>)
        outs(%init : tensor<2x3xbf16>)
        attrs = {merlin.numeric_semantics = "ordered-fp8-bf16-example-v1",
                 merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
                 merlin.reduction_order = "k-ascending"} {
        ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
          %ab = arith.extf %av : f8E4M3FN to bf16
          %wb = arith.extf %wv : f8E4M3FN to bf16
          %next = math.fma %ab, %wb, %acc : bf16
          linalg.yield %next : bf16
      } -> tensor<2x3xbf16>
      func.return %result : tensor<2x3xbf16>
    } }"""


def _ordered_fp8_pack_chain_source() -> str:
    first = _ordered_fp8_source()
    old_signature = (
        '%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})\n'
        '      -> tensor<2x3xbf16>'
    )
    new_signature = (
        '%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},\n'
        '      %next_w: tensor<4x3xf8E4M3FN> '
        '{merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})\n'
        '      -> tensor<2x4xbf16>'
    )
    assert old_signature in first
    first = first.replace(old_signature, new_signature, 1)
    old_return = 'func.return %result : tensor<2x3xbf16>'
    assert old_return in first
    return first.replace(old_return, """
      %pack_empty = tensor.empty() : tensor<2x3xf8E4M3FN>
      %packed = linalg.generic {
        indexing_maps = [affine_map<(m,k)->(m,k)>, affine_map<(m,k)->(m,k)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%result : tensor<2x3xbf16>)
        outs(%pack_empty : tensor<2x3xf8E4M3FN>)
        attrs = {merlin.numeric_semantics = "declared-exact-fp8-policy",
                 merlin.input_domain = "exact-seven-code-domain",
                 merlin.scale_code = 127 : i32} {
        ^bb0(%value: bf16, %old: f8E4M3FN):
          %converted = arith.truncf %value : bf16 to f8E4M3FN
          linalg.yield %converted : f8E4M3FN
      } -> tensor<2x3xf8E4M3FN>
      %zero2 = arith.constant 0.000000e+00 : bf16
      %empty2 = tensor.empty() : tensor<2x4xbf16>
      %init2 = linalg.fill ins(%zero2 : bf16) outs(%empty2 : tensor<2x4xbf16>)
        -> tensor<2x4xbf16>
      %second = linalg.generic {
        indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                         affine_map<(m,n,k)->(n,k)>,
                         affine_map<(m,n,k)->(m,n)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%packed, %next_w : tensor<2x3xf8E4M3FN>, tensor<4x3xf8E4M3FN>)
        outs(%init2 : tensor<2x4xbf16>)
        attrs = {merlin.numeric_semantics = "ordered-fp8-bf16-example-v1",
                 merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
                 merlin.reduction_order = "k-ascending"} {
        ^bb0(%av: f8E4M3FN, %wv: f8E4M3FN, %acc: bf16):
          %ab = arith.extf %av : f8E4M3FN to bf16
          %wb = arith.extf %wv : f8E4M3FN to bf16
          %next = math.fma %ab, %wb, %acc : bf16
          linalg.yield %next : bf16
      } -> tensor<2x4xbf16>
      func.return %second : tensor<2x4xbf16>""", 1)


def _ordered_fp8_k2_source() -> str:
    first = _ordered_fp8_source()
    old_signature = (
        '%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})\n'
        '      -> tensor<2x3xbf16>'
    )
    new_signature = (
        '%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},\n'
        '      %a1: tensor<2x4xf8E4M3FN> '
        '{merlin.input_domain = "finite-normal-e4m3-or-positive-zero"},\n'
        '      %w1: tensor<3x4xf8E4M3FN> '
        '{merlin.input_domain = "finite-normal-e4m3-or-positive-zero"})\n'
        '      -> tensor<2x3xbf16>'
    )
    assert old_signature in first
    first = first.replace(old_signature, new_signature, 1)
    old_return = 'func.return %result : tensor<2x3xbf16>'
    assert old_return in first
    return first.replace(old_return, """
      %continued = linalg.generic {
        indexing_maps = [affine_map<(m,n,k)->(m,k)>,
                         affine_map<(m,n,k)->(n,k)>,
                         affine_map<(m,n,k)->(m,n)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%a1, %w1 : tensor<2x4xf8E4M3FN>, tensor<3x4xf8E4M3FN>)
        outs(%result : tensor<2x3xbf16>)
        attrs = {merlin.numeric_semantics = "ordered-fp8-bf16-example-v1",
                 merlin.input_domain = "finite-normal-e4m3-or-positive-zero",
                 merlin.reduction_order = "k-ascending"} {
        ^bb0(%av1: f8E4M3FN, %wv1: f8E4M3FN, %acc1: bf16):
          %ab1 = arith.extf %av1 : f8E4M3FN to bf16
          %wb1 = arith.extf %wv1 : f8E4M3FN to bf16
          %next1 = math.fma %ab1, %wb1, %acc1 : bf16
          linalg.yield %next1 : bf16
      } -> tensor<2x3xbf16>
      func.return %continued : tensor<2x3xbf16>""", 1)


def test_ordered_fp8_k2_keeps_accumulator_dependency_and_policy() -> None:
    source = _ordered_fp8_k2_source()
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-fp8",
                                       lowering_policy="diagnostic")
    request = translated.request
    assert tuple(node.op for node in request.nodes) == (
        "input", "input", "input", "input", "matmul", "matmul_accumulate"
    )
    reset, continued = request.nodes[-2:]
    assert continued.inputs == (request.nodes[2].id, request.nodes[3].id, reset.id)
    assert continued.type == reset.type == TensorType((2, 3), "bf16", "ordered-fp8-bf16-example-v1")
    assert tuple(index_map.apply((1, 2, 3)) for index_map in continued.index_maps) == (
        (1, 3), (2, 3), (1, 2), (1, 2)
    )
    assert request.outputs == (continued.id,)
    assert KernelRequest.from_record(request.record()).digest() == request.digest()


def test_ordered_fp8_k2_rejects_changed_accumulator_or_fma() -> None:
    source = _ordered_fp8_k2_source()
    for changed, reason in (
        (source.replace('outs(%result : tensor<2x3xbf16>)',
                        'outs(%empty : tensor<2x3xbf16>)'), "matching prior ordered accumulator"),
        (source.replace('merlin.numeric_semantics = "ordered-fp8-bf16-example-v1"',
                        'merlin.numeric_semantics = "different-order-v1"', 1),
         "matching prior ordered accumulator"),
        (source.replace('%next1 = math.fma %ab1, %wb1, %acc1 : bf16',
                        '%next1 = arith.addf %ab1, %acc1 : bf16'), "extf, extf, fma, yield"),
    ):
        assert changed != source
        with pytest.raises(LinalgBridgeError, match=reason):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-fp8",
                                  lowering_policy="diagnostic")


def test_parsed_ordered_fp8_pack_chain_preserves_computed_operand() -> None:
    source = _ordered_fp8_pack_chain_source()
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-fp8",
                                       lowering_policy="diagnostic")
    nodes = translated.request.nodes
    assert tuple(node.op for node in nodes) == (
        "input", "input", "input", "matmul", "pack_fp8", "matmul"
    )
    assert nodes[4].inputs == (nodes[3].id,)
    assert nodes[4].type.numerical_policy == "declared-exact-fp8-policy"
    assert nodes[5].inputs == (nodes[4].id, nodes[2].id)
    assert nodes[5].type.shape == (2, 4)
    assert translated.request.outputs == (nodes[5].id,)
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()

    for changed, reason in (
        (source.replace('"declared-exact-fp8-policy"', '"raw-bit-pattern"'), "computed FP8 pack"),
        (source.replace('%next_w: tensor<4x3xf8E4M3FN> '
                        '{merlin.input_domain = "finite-normal-e4m3-or-positive-zero"}',
                        '%next_w: tensor<4x3xf8E4M3FN> '
                        '{merlin.input_domain = "all-bit-patterns"}'), "declared boundary"),
        (source.replace('affine_map<(m,n,k)->(n,k)>', 'affine_map<(m,n,k)->(k,n)>'), "maps"),
    ):
        assert changed != source
        with pytest.raises(LinalgBridgeError, match=reason):
            translate_linalg_text(changed, entry="work", target_identity="synthetic-fp8",
                                  lowering_policy="diagnostic")


def test_ordered_fp8_bf16_linalg_preserves_policy_orientation_and_zero_init() -> None:
    source = _ordered_fp8_source()
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-fp8",
                                       lowering_policy="diagnostic")
    request = translated.request
    assert translated.source_operations == (
        "arith.constant", "tensor.empty", "linalg.fill", "linalg.generic", "func.return"
    )
    assert request.source_identity == sha256(source.encode()).hexdigest()
    assert tuple(node.op for node in request.nodes) == ("input", "input", "matmul")
    assert request.nodes[-1].type == TensorType((2, 3), "bf16", "ordered-fp8-bf16-example-v1")
    assert tuple(index_map.apply((1, 2, 3)) for index_map in request.nodes[-1].index_maps) == (
        (1, 3), (2, 3), (1, 2)
    )
    assert request.nodes[0].attrs == request.nodes[1].attrs == (
        ("input_domain", "finite-normal-e4m3-or-positive-zero"),
    )
    assert not request.constants
    assert KernelRequest.from_record(request.record()).digest() == request.digest()


@pytest.mark.parametrize("before,after", [
    ("arith.constant 0.000000e+00", "arith.constant -0.000000e+00"),
    ("affine_map<(m,n,k)->(n,k)>", "affine_map<(m,n,k)->(k,n)>"),
    ("math.fma %ab, %wb, %acc", "math.fma %ab, %wb, %ab"),
    ("math.fma %ab, %wb, %acc : bf16", "math.fma %ab, %wb, %acc fastmath<reassoc> : bf16"),
    ("merlin.reduction_order = \"k-ascending\"", "merlin.reduction_order = \"k-descending\""),
    ("%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = \"finite-normal-e4m3-or-positive-zero\"}",
     "%w: tensor<3x4xf8E4M3FN> {merlin.input_domain = \"all-bit-patterns\"}"),
])
def test_ordered_fp8_bf16_linalg_rejects_changed_premises(before: str, after: str) -> None:
    source = _ordered_fp8_source().replace(before, after, 1)
    assert source != _ordered_fp8_source()
    with pytest.raises(LinalgBridgeError):
        translate_linalg_text(source, entry="work", target_identity="synthetic-fp8",
                              lowering_policy="diagnostic")


def test_fp8_transpose_preserves_all_bytes_and_rejects_arithmetic() -> None:
    source = """module { func.func @work(%a: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf8E4M3FN> {
      %empty = tensor.empty() : tensor<32x32xf8E4M3FN>
      %result = linalg.transpose ins(%a : tensor<32x32xf8E4M3FN>)
        outs(%empty : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
      func.return %result : tensor<32x32xf8E4M3FN>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-fp8-movement")
    tensor = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    bits = tuple(range(256)) * 4
    result = evaluate_graph(translated.request, {"arg0": TensorValue(tensor, bits)})[0]
    assert result.type == tensor
    assert result.elements == tuple(bits[32 * row + col] for col in range(32) for row in range(32))
    assert translated.request.nodes[-1].op == "transpose"
    assert KernelRequest.from_record(translated.request.record()).digest() == translated.request.digest()

    arithmetic = source.replace(
        "func.return %result : tensor<32x32xf8E4M3FN>",
        "%sum = arith.addf %result, %result : tensor<32x32xf8E4M3FN>\n"
        "      func.return %sum : tensor<32x32xf8E4M3FN>",
    )
    with pytest.raises(LinalgBridgeError, match="no admitted translation"):
        translate_linalg_text(arithmetic, entry="work", target_identity="synthetic-fp8-movement")


@pytest.mark.parametrize("mutation", ["unsigned", "dropped_accumulator"])
def test_parsed_generic_rejects_changed_scalar_semantics(mutation: str) -> None:
    with pytest.raises(LinalgBridgeError, match="body|widen|yield"):
        translate_linalg_text(_generic_source(body=mutation), entry="work", target_identity="synthetic")


def test_parsed_generic_rejects_changed_indexing_map() -> None:
    with pytest.raises(LinalgBridgeError, match="indexing maps"):
        translate_linalg_text(_generic_source(swapped_map=True), entry="work", target_identity="synthetic")


def test_parsed_return_must_match_declared_output_types() -> None:
    source = _generic_source(two_outputs=True).replace(
        "-> (tensor<2x2xi32>, tensor<2x2xi32>)",
        "-> (tensor<2x2xi32>, tensor<1x2xi32>)",
    )
    with pytest.raises(LinalgBridgeError, match="entry signature"):
        translate_linalg_text(source, entry="work", target_identity="synthetic")


def test_named_i32_matmul_uses_same_initialized_graph_primitive() -> None:
    source = """module { func.func @work(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
      %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %r = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
        outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    tensor = TensorType((2, 2), "i32", "i32-wrap-k-ascending")
    values = {
        "arg0": TensorValue(tensor, (1, 2, 3, 4)),
        "arg1": TensorValue(tensor, (5, 6, 7, 8)),
        "arg2": TensorValue(tensor, (10, 20, 30, 40)),
    }
    assert evaluate_graph(translated.request, values)[0].elements == (29, 42, 73, 90)


def test_parsed_transpose_preserves_index_map_and_reaches_native_selection(bridge: Path) -> None:
    source = """module { func.func @work(%a: tensor<2x3xi8>) -> tensor<3x2xi8> {
      %empty = tensor.empty() : tensor<3x2xi8>
      %result = linalg.transpose ins(%a : tensor<2x3xi8>)
        outs(%empty : tensor<3x2xi8>) permutation = [1, 0]
      func.return %result : tensor<3x2xi8>
    } }"""
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic-transpose-v1")
    assert translated.source_operations == ("tensor.empty", "linalg.transpose", "func.return")
    assert KernelRequest.from_record(translated.request.record()).record() == translated.request.record()
    value = TensorValue(TensorType((2, 3), "i8", "signed-i8"), (1, 2, 3, 4, 5, 6))
    assert evaluate_graph(translated.request, {"arg0": value})[0].elements == (1, 4, 2, 5, 3, 6)

    node = translated.request.nodes[-1]
    descriptor = InstructionDescriptor(
        "synthetic_transpose",
        "transpose",
        ("external",),
        "external",
        "i8",
        "signed-i8",
        (2,),
        input_dtypes=("i8",),
        input_numerical_policies=("signed-i8",),
        input_ranks=(2,),
        index_maps=node.index_maps,
        shape_contract="relations",
        shape_equalities=(AxisEquality("in0", 0, "out", 1), AxisEquality("in0", 1, "out", 0)),
    )
    selected = select_and_allocate(
        translated.request,
        (descriptor,),
        (StorageBank("external", "dram", 3, "word"),),
        bridge=bridge,
        fixed_inputs={"arg0": 0},
        fixed_outputs=(1,),
    )
    assert selected.status == "selected", selected.reason
    assert selected.candidate is not None and selected.candidate.instruction_count() == 1
    bad_node = replace(node, index_maps=tuple(reversed(node.index_maps)))
    bad_request = replace(translated.request, nodes=(*translated.request.nodes[:-1], bad_node))
    with pytest.raises(ValueError, match="transpose"):
        evaluate_graph(bad_request, {"arg0": value})
    assert not descriptor.accepts(bad_node, (translated.request.nodes[0],))


def test_parsed_transpose_admits_identity_and_rejects_invalid_permutation_or_input() -> None:
    source = """module { func.func @work(%a: tensor<2x3xi8>) -> tensor<3x2xi8> {
      %empty = tensor.empty() : tensor<3x2xi8>
      %result = linalg.transpose ins(%a : tensor<2x3xi8>)
        outs(%empty : tensor<3x2xi8>) permutation = [1, 0]
      func.return %result : tensor<3x2xi8>
    } }"""
    identity = source.replace("tensor<3x2xi8>", "tensor<2x3xi8>").replace(
        "permutation = [1, 0]", "permutation = [0, 1]"
    )
    identity_request = translate_linalg_text(identity, entry="work", target_identity="synthetic").request
    assert identity_request.nodes[-1].index_maps == pointwise_identity_maps(2, 1)
    invalid = source.replace("permutation = [1, 0]", "permutation = [0, 0]")
    with pytest.raises(LinalgBridgeError, match="permutation"):
        translate_linalg_text(invalid, entry="work", target_identity="synthetic")
    uninitialized = source.replace("%a : tensor<2x3xi8>", "%source : tensor<2x3xi8>").replace(
        "%empty = tensor.empty()", "%source = tensor.empty() : tensor<2x3xi8>\n      %empty = tensor.empty()"
    )
    with pytest.raises(LinalgBridgeError, match="uninitialized"):
        translate_linalg_text(uninitialized, entry="work", target_identity="synthetic")


def test_parsed_region_reaches_native_rule_extraction_and_allocation(bridge: Path) -> None:
    translated = translate_linalg_text(_generic_source(), entry="work", target_identity="synthetic")
    maps = translated.request.nodes[-1].index_maps
    descriptor = InstructionDescriptor(
        "synthetic_contract",
        "matmul_accumulate",
        ("external",) * 3,
        "external",
        "i32",
        "i32-wrap-k-ascending",
        (2,),
        input_dtypes=("i8", "i8", "i32"),
        input_numerical_policies=("signed-i8", "signed-i8", "i32-wrap-k-ascending"),
        input_ranks=(2, 2, 2),
        index_maps=maps,
        shape_contract="relations",
        shape_equalities=(
            AxisEquality("in0", 0, "out", 0),
            AxisEquality("in0", 1, "in1", 0),
            AxisEquality("in1", 1, "out", 1),
            AxisEquality("in2", 0, "out", 0),
            AxisEquality("in2", 1, "out", 1),
        ),
    )
    source = translated.request.nodes[-1]
    inputs = tuple(translated.request.node(child) for child in source.inputs)
    assert descriptor.accepts(source, inputs)
    assert not descriptor.accepts(
        source,
        (inputs[0], replace(inputs[1], type=TensorType((4, 2), "i8", "signed-i8")), inputs[2]),
    )
    altered_nodes = tuple(
        replace(node, type=TensorType((4, 2), "i8", "signed-i8")) if node.id == "arg1" else node
        for node in translated.request.nodes
    )
    assert not generate_rules(replace(translated.request, nodes=altered_nodes), (descriptor,)).rewrites
    selected = select_and_allocate(
        translated.request,
        (descriptor,),
        (StorageBank("external", "dram", 4, "word"),),
        bridge=bridge,
        fixed_inputs={"arg0": 0, "arg1": 1, "arg2": 2},
    )
    assert selected.status == "selected", selected.reason
    assert selected.engine == "merlin_native" and selected.check_fingerprint
    assert selected.graph is not None and selected.graph.value(selected.graph.outputs[0]).symbol.startswith("i_")
