"""Mask comparison and projection preserve exact source predicates."""

from dataclasses import replace

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import IndexMap
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source(dtype, operation):
    return f"""module {{
      func.func @forward(%a: tensor<2x3x{dtype}>, %b: tensor<1x3x{dtype}>) -> tensor<2x3xi1> {{
        %empty = tensor.empty() : tensor<2x3xi1>
        %result = linalg.generic {{indexing_maps = [affine_map<(i,j)->(i,j)>,
          affine_map<(i,j)->(0,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}}
          ins(%a, %b : tensor<2x3x{dtype}>, tensor<1x3x{dtype}>) outs(%empty : tensor<2x3xi1>) {{
          ^bb0(%lhs: {dtype}, %rhs: {dtype}, %unused: i1):
            %value = {operation} %lhs, %rhs : {dtype}
            linalg.yield %value : i1
          }} -> tensor<2x3xi1>
        return %result : tensor<2x3xi1>
      }}
    }}"""


def test_boolean_widening_keeps_true_as_one_and_rejects_signed_extension():
    text = """module {
      func.func @forward(%input: tensor<4xi1>) -> tensor<4xi64> {
        %empty = tensor.empty() : tensor<4xi64>
        %result = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
          iterator_types = ["parallel"]} ins(%input : tensor<4xi1>) outs(%empty : tensor<4xi64>) {
          ^bb0(%value: i1, %unused: i64):
            %wide = arith.extui %value : i1 to i64
            linalg.yield %wide : i64
        } -> tensor<4xi64>
        return %result : tensor<4xi64>
      }
    }"""
    request = translate_linalg_text(text, entry="forward", target_identity="synthetic").request
    inputs = {"arg0": TensorValue(request.node("arg0").type, (0, 1, 1, 0))}
    assert evaluate_graph(request, inputs)[0].elements == (0, 1, 1, 0)
    with pytest.raises(ValueError):
        translate_linalg_text(text.replace("arith.extui", "arith.extsi"), entry="forward", target_identity="synthetic")


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("slt", (1, 0, 0, 0, 1, 1)),
        ("sle", (1, 1, 1, 0, 1, 1)),
        ("sgt", (0, 0, 0, 1, 0, 0)),
        ("sge", (0, 1, 1, 1, 0, 0)),
        ("eq", (0, 1, 1, 0, 0, 0)),
        ("ne", (1, 0, 0, 1, 1, 1)),
    ],
)
def test_signed_i64_comparison_masks_and_projected_inputs(predicate, expected):
    translated = translate_linalg_text(
        source("i64", "arith.cmpi " + predicate + ","), entry="forward", target_identity="synthetic"
    )
    request = translated.request
    inputs = {
        "arg0": TensorValue(request.node("arg0").type, (-(1 << 63), 0, (1 << 63) - 1, 1, -1, 3)),
        "arg1": TensorValue(request.node("arg1").type, (0, 0, (1 << 63) - 1)),
    }
    assert evaluate_graph(request, inputs)[0].elements == expected
    bad = replace(
        request.nodes[-1],
        index_maps=(
            request.nodes[-1].index_maps[0],
            IndexMap(2, ((0, 0), (0, 1)), (1, 0)),
            request.nodes[-1].index_maps[-1],
        ),
    )
    with pytest.raises(ValueError, match="projection"):
        evaluate_graph(replace(request, nodes=(*request.nodes[:-1], bad)), inputs)


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("oeq", (1, 0, 1, 1, 1, 1)),
        ("one", (0, 0, 0, 0, 0, 0)),
        ("olt", (0, 0, 0, 0, 0, 0)),
        ("ole", (1, 0, 1, 1, 1, 1)),
        ("ogt", (0, 0, 0, 0, 0, 0)),
        ("oge", (1, 0, 1, 1, 1, 1)),
    ],
)
def test_ordered_float_masks_do_not_turn_nan_into_true(predicate, expected):
    request = translate_linalg_text(
        source("f32", "arith.cmpf " + predicate + ","), entry="forward", target_identity="synthetic"
    ).request
    inputs = {
        "arg0": TensorValue(request.node("arg0").type, (0x80000000, 0x7FC00000, 0x7F800000, 0, 0x3F800000, 0x7F800000)),
        "arg1": TensorValue(request.node("arg1").type, (0, 0x3F800000, 0x7F800000)),
    }
    assert evaluate_graph(request, inputs)[0].elements == expected


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("arith.andi", (0, 0, 1, 1, 0, 0)),
        ("arith.xori", (1, 1, 0, 0, 0, 1)),
    ],
)
def test_boolean_mask_operations(operation, expected):
    request = translate_linalg_text(source("i1", operation), entry="forward", target_identity="synthetic").request
    inputs = {
        "arg0": TensorValue(request.node("arg0").type, (0, 1, 1, 1, 0, 0)),
        "arg1": TensorValue(request.node("arg1").type, (1, 0, 1)),
    }
    assert evaluate_graph(request, inputs)[0].elements == expected
