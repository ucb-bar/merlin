"""Splat literals read only by elementwise nodes stay one element."""

from struct import pack, unpack

from merlin.compiler.lower.c.float_arithmetic import emit_float_arithmetic_c
from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.model import IndexMap, KernelRequest
from merlin.compiler.semantic.reference import TensorValue, evaluate_graph


def _scaled(returned="%result", types="tensor<2x3x4xf32>"):
    return f'''module {{
func.func @work(%x: tensor<2x3x4xf32>) -> ({types}) {{
  %c = arith.constant 1.250000e-01 : f32
  %scale = tensor.splat %c : tensor<2x3x4xf32>
  %empty = tensor.empty() : tensor<2x3x4xf32>
  %result = linalg.generic {{
    indexing_maps = [affine_map<(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)>, affine_map<(i,j,k)->(i,j,k)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  }} ins(%x, %scale : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%empty : tensor<2x3x4xf32>) {{
    ^bb0(%a: f32, %b: f32, %old: f32):
      %value = arith.mulf %a, %b : f32
      linalg.yield %value : f32
  }} -> tensor<2x3x4xf32>
  func.return {returned} : {types}
}}
}}'''


def _bits(value):
    return unpack("<I", pack("<f", value))[0]


def test_splat_read_by_elementwise_multiply_is_one_element():
    translated = translate_linalg_text(_scaled(), entry="work", target_identity="synthetic")
    request = translated.request
    (binding,) = request.constants
    literal = request.node(binding.node_id)
    assert (binding.encoding, binding.data_hex) == ("f32-le-raw", "0000003e")
    assert literal.type.shape == (1, 1, 1)
    product = request.node(request.outputs[0])
    assert product.op == "f32_mul" and product.inputs[1] == literal.id
    assert product.index_maps[1] == IndexMap(3, ((0, 0, 0),) * 3, (0, 0, 0))
    assert KernelRequest.from_record(request.record()).digest() == request.digest()
    # Nonzero inputs; scaling by a power of two is exact in binary32 here.
    values = [(-1) ** i * (i + 1) * 1.5 for i in range(24)]
    (result,) = evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, tuple(map(_bits, values)))},
                               constants=translated.constants)
    assert result.elements == tuple(_bits(value * 0.125) for value in values)
    record, _ = emit_float_arithmetic_c(request, "scale")
    assert record["inputs"] == ["arg0", literal.id]


def test_returned_splat_stays_dense():
    translated = translate_linalg_text(_scaled("%result, %scale", "tensor<2x3x4xf32>, tensor<2x3x4xf32>"),
                                       entry="work", target_identity="synthetic")
    (binding,) = translated.request.constants
    assert binding.data_hex == "0000003e" * 24
    assert translated.request.node(binding.node_id).type.shape == (2, 3, 4)


def test_zero_splat_padded_at_offset_zero_stays_dense():
    # Asymmetric zero padding: the insertion starts at offset zero, so its placement equals the identity.
    source = '''module {
func.func @work(%x: tensor<1x2x3x3xbf16>) -> tensor<1x2x4x4xbf16> {
  %c = arith.constant 0.000000e+00 : bf16
  %zero = tensor.splat %c : tensor<1x2x4x4xbf16>
  %padded = "tensor.insert_slice"(%x, %zero) <{static_offsets = array<i64: 0, 0, 0, 0>,
    static_sizes = array<i64: 1, 2, 3, 3>, static_strides = array<i64: 1, 1, 1, 1>,
    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<1x2x3x3xbf16>, tensor<1x2x4x4xbf16>) -> tensor<1x2x4x4xbf16>
  func.return %padded : tensor<1x2x4x4xbf16>
}
}'''
    translated = translate_linalg_text(source, entry="work", target_identity="synthetic")
    request = translated.request
    padded = request.node(request.outputs[0])
    destination = request.node(padded.inputs[1])
    assert padded.op == "insert_slice" and destination.type.shape == (1, 2, 4, 4)
    assert padded.index_maps[1] == padded.index_maps[2]
    values = tuple(0x3f80 + 3 * i for i in range(18))  # nonzero BF16 bits
    (result,) = evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, values)},
                               constants=translated.constants)
    expected = [0] * 32
    for i in range(18):
        channel, row, column = i // 9, i // 3 % 3, i % 3
        expected[channel * 16 + row * 4 + column] = values[i]
    assert result.elements == tuple(expected)
