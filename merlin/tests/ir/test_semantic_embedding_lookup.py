"""Parsed i64 table lookup keeps runtime indices and exact floating element bits."""

from __future__ import annotations

import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

SOURCE = """module {
  func.func @forward(%table: tensor<3x2xf32>, %ids: tensor<2x2xi64>) -> tensor<2x2x2xf32> {
    %empty = tensor.empty() : tensor<2x2x2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2)->(d0,d1)>,
                       affine_map<(d0,d1,d2)->(d0,d1,d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%ids : tensor<2x2xi64>) outs(%empty : tensor<2x2x2xf32>) {
    ^bb0(%id: i64, %old: f32):
      %row = arith.index_cast %id : i64 to index
      %column = linalg.index 2 : index
      %value = tensor.extract %table[%row, %column] : tensor<3x2xf32>
      linalg.yield %value : f32
    } -> tensor<2x2x2xf32>
    return %result : tensor<2x2x2xf32>
  }
}
"""


def _request(source: str = SOURCE, *, constants: dict[int, bytes] | None = None):
    return translate_linalg_text(
        source, entry="forward", target_identity="synthetic", argument_constants=constants,
    )


def test_runtime_embedding_preserves_exact_bits_and_round_trips():
    translated = _request()
    request = translated.request
    lookup = request.nodes[-1]
    assert lookup.op == "embedding_lookup"
    assert lookup.inputs == ("arg1", "arg0")
    assert dict(lookup.attrs) == {
        "index_domain": "nonnegative-below-table-rows", "minimum_index_bits": 3,
    }
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    table = TensorValue(request.node("arg0").type, (
        0x80000000, 0x7FC00001, 0x3F800000, 0x00000000, 0xBF800000, 0x7F800000,
    ))
    ids = TensorValue(request.node("arg1").type, (2, 0, 2, 1))
    output, = evaluate_graph(request, {"arg0": table, "arg1": ids})
    assert output.elements == (
        0xBF800000, 0x7F800000, 0x80000000, 0x7FC00001,
        0xBF800000, 0x7F800000, 0x3F800000, 0x00000000,
    )


@pytest.mark.parametrize(
    "dtype,bits",
    (
        ("bf16", (0x8000, 0x7FC1, 0x3F80, 0, 0xBF80, 0x7F80)),
        ("f64", (1 << 63, 0x7FF8000000000001, 0x3FF0000000000000, 0, 0xBFF0000000000000, 0x7FF0000000000000)),
        ("f8E4M3FN", (0x80, 0x7F, 0x38, 0, 0xB8, 0x7E)),
    ),
)
def test_other_float_encodings_are_bit_preserving(dtype, bits):
    translated = _request(SOURCE.replace("f32", dtype))
    graph = translated.request
    table = TensorValue(graph.node("arg0").type, bits)
    ids = TensorValue(graph.node("arg1").type, (2, 0, 2, 1))
    (output,) = evaluate_graph(graph, {"arg0": table, "arg1": ids})
    assert output.elements == bits[4:6] + bits[:2] + bits[4:6] + bits[2:4]


def test_declared_i64_constant_is_checked_against_its_exact_bytes():
    data = b"".join(index.to_bytes(8, "little", signed=True) for index in (2, 0, 2, 1))
    translated = _request(constants={1: data})
    request = translated.request
    assert request.constants[0].encoding == "i64-le"
    assert request.constants[0].data_hex == data.hex()
    with pytest.raises(ValueError, match="signed i64 constant"):
        request.constants[0].validate_type(TensorType((2, 2), "i64", "raw-bit-pattern"))
    table = TensorValue(request.node("arg0").type, (10, 11, 20, 21, 30, 31))
    output, = evaluate_graph(request, {"arg0": table}, constants=translated.constants)
    assert output.elements == (30, 31, 10, 11, 30, 31, 20, 21)
    with pytest.raises(ValueError, match="reference constant differs"):
        evaluate_graph(request, {"arg0": table}, constants={
            "arg1": TensorValue(request.node("arg1").type, (1, 0, 2, 1)),
        })


@pytest.mark.parametrize("index", [-1, 3])
def test_runtime_index_outside_table_refuses(index: int):
    translated = _request()
    request = translated.request
    table = TensorValue(request.node("arg0").type, (1, 2, 3, 4, 5, 6))
    ids = TensorValue(request.node("arg1").type, (0, index, 1, 2))
    with pytest.raises(ValueError, match="embedding index is outside"):
        evaluate_graph(request, {"arg0": table, "arg1": ids})


def test_signed_i64_boundary_refuses_unrepresentable_value():
    request = _request().request
    with pytest.raises(ValueError, match="signed-i64 value is outside"):
        TensorValue(request.node("arg1").type, (0, 1, 2, 1 << 63))


@pytest.mark.parametrize(
    "producer",
    [
        "%result = tensor.splat %value : tensor<2xi64>",
        "%empty = tensor.empty() : tensor<2xi64>\n"
        "    %result = linalg.fill ins(%value : i64) outs(%empty : tensor<2xi64>) -> tensor<2xi64>",
    ],
)
def test_i64_scalar_splat_and_fill_are_initialized(producer: str):
    source = (
        "module {\n"
        "  func.func @forward() -> tensor<2xi64> {\n"
        "    %value = arith.constant -3 : i64\n"
        f"    {producer}\n"
        "    return %result : tensor<2xi64>\n"
        "  }\n"
        "}\n"
    )
    translated = _request(source)
    request = translated.request
    assert request.constants[0].encoding == "i64-le"
    assert evaluate_graph(request, {}, constants=translated.constants)[0].elements == (-3, -3)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("(d0,d1)>,", "(d1,d0)>,"),
        ("linalg.index 2 : index", "linalg.index 1 : index"),
        ("%table[%row, %column]", "%table[%column, %row]"),
        ("linalg.yield %value : f32", "linalg.yield %old : f32"),
        ("arith.index_cast %id : i64 to index", "arith.index_cast %id : i64 to i32"),
    ],
)
def test_embedding_source_mutations_refuse(old: str, new: str):
    with pytest.raises((LinalgBridgeError, ValueError)):
        _request(SOURCE.replace(old, new))


def test_multiple_consumers_of_indices_are_preserved():
    source = SOURCE.replace(
        "-> tensor<2x2x2xf32> {", "-> (tensor<2x2x2xf32>, tensor<2x2xi64>) {",
    ).replace(
        "return %result : tensor<2x2x2xf32>",
        "return %result, %ids : tensor<2x2x2xf32>, tensor<2x2xi64>",
    )
    translated = _request(source)
    request = translated.request
    assert request.outputs == (request.nodes[-1].id, "arg1")
    table = TensorValue(request.node("arg0").type, (1, 2, 3, 4, 5, 6))
    ids = TensorValue(request.node("arg1").type, (2, 0, 2, 1))
    _, retained_ids = evaluate_graph(request, {"arg0": table, "arg1": ids})
    assert retained_ids == ids
