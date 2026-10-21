"""Typed views preserve every element bit; physical placement stays separate."""

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source(dtype):
    return f"""module {{
      func.func @forward(%input: tensor<2x1x2x{dtype}>) -> tensor<2x3x2x{dtype}> {{
        %empty = tensor.empty() : tensor<2x3x2x{dtype}>
        %result = linalg.generic {{indexing_maps = [affine_map<(a,b,c)->(a,0,c)>,
          affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel", "parallel", "parallel"]}}
          ins(%input : tensor<2x1x2x{dtype}>) outs(%empty : tensor<2x3x2x{dtype}>) {{
        ^bb0(%value: {dtype}, %unused: {dtype}):
          linalg.yield %value : {dtype}
        }} -> tensor<2x3x2x{dtype}>
        return %result : tensor<2x3x2x{dtype}>
      }}
    }}"""


def translate(text):
    return translate_linalg_text(text, entry="forward", target_identity="synthetic")


@pytest.mark.parametrize(
    "dtype,values",
    (
        ("bf16", (0x8000, 0x7FC1, 1, 0x7F80)),
        ("f8E4M3FN", (0x80, 0x7F, 1, 0x7E)),
        ("i8", (-128, 127, -1, 0)),
        ("i32", (-(1 << 31), (1 << 31) - 1, -1, 0)),
    ),
)
def test_broadcast_is_a_bit_preserving_affine_read(dtype, values):
    graph = translate(source(dtype)).request
    (output,) = evaluate_graph(graph, {"arg0": TensorValue(graph.node("arg0").type, values)})
    assert output.elements == values[:2] * 3 + values[2:] * 3
    assert len(graph.nodes) == 2 and graph.nodes[-1].op == "gather"
    assert graph.nodes[-1].inputs == ("arg0",)
    assert graph.nodes[-1].type.numerical_policy == graph.node("arg0").type.numerical_policy
    assert type(graph).from_record(graph.record()).digest() == graph.digest()


@pytest.mark.parametrize(
    "before,after",
    (
        ("(a,0,c)", "(a,1,c)"),
        ("(a,b,c)>],", "(a,c,b)>],"),
        ("linalg.yield %value", "linalg.yield %unused"),
        ('"parallel", "parallel", "parallel"', '"parallel", "reduction", "parallel"'),
    ),
)
def test_changed_bounds_coverage_or_source_cannot_enter_as_a_view(before, after):
    with pytest.raises(ValueError):
        translate(source("bf16").replace(before, after))
