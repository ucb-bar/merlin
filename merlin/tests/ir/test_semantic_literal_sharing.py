"""Repeated immutable splats share bytes while output order remains explicit."""

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import evaluate_graph


@pytest.mark.parametrize(
    "dtype,literal,value,width",
    (
        ("f32", "-0.0", 0x80000000, 4),
        ("f64", "-0.0", 1 << 63, 8),
        ("i1", "true", 1, 1),
        ("i32", "-5", -5, 4),
        ("i64", "-3", -3, 8),
    ),
)
def test_identical_splats_and_fills_share_one_exact_literal(dtype, literal, value, width):
    typ = f"tensor<4x4x{dtype}>"
    operations = []
    for index in range(16):
        typed_literal = literal if dtype == "i1" else f"{literal} : {dtype}"
        operations.append(f"%scalar{index} = arith.constant {typed_literal}")
        if index % 2 and dtype in {"f32", "i32", "i64"}:
            operations.append(f"%empty{index} = tensor.empty() : {typ}")
            operations.append(
                f"%tensor{index} = linalg.fill ins(%scalar{index} : {dtype}) outs(%empty{index} : {typ}) -> {typ}"
            )
        else:
            operations.append(f"%tensor{index} = tensor.splat %scalar{index} : {typ}")
    source = f"module {{ func.func @forward() -> ({typ}, {typ}) {{ " + "\n".join(operations)
    source += f"\nreturn %tensor15, %tensor0 : {typ}, {typ} }} }}"
    translated = translate_linalg_text(source, entry="forward", target_identity="synthetic")
    graph = translated.request
    outputs = evaluate_graph(graph, {}, constants=translated.constants)
    assert [output.elements for output in outputs] == [(value,) * 16] * 2
    assert len(graph.constants) == len(translated.constants) == len(graph.nodes) == 1
    assert len(graph.constants[0].data_hex) == width * 16 * 2
    assert graph.outputs == (graph.nodes[0].id,) * 2


def test_float_signs_shapes_and_storage_types_are_not_merged():
    source = """module {
      func.func @forward() -> (tensor<4xf32>, tensor<2x2xf32>, tensor<4xf32>, tensor<4xf64>) {
        %plus = arith.constant 0.0 : f32
        %minus = arith.constant -0.0 : f32
        %double = arith.constant 0.0 : f64
        %a = tensor.splat %plus : tensor<4xf32>
        %b = tensor.splat %minus : tensor<4xf32>
        %c = tensor.splat %plus : tensor<2x2xf32>
        %d = tensor.splat %double : tensor<4xf64>
        return %b, %c, %a, %d : tensor<4xf32>, tensor<2x2xf32>, tensor<4xf32>, tensor<4xf64>
      }
    }"""
    translated = translate_linalg_text(source, entry="forward", target_identity="synthetic")
    assert len(translated.request.constants) == 4
    result = evaluate_graph(translated.request, {}, constants=translated.constants)
    assert [tensor.elements for tensor in result] == [(0x80000000,) * 4, (0,) * 4, (0,) * 4, (0,) * 4]
