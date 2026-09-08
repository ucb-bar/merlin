"""Regression gates for exact per-tensor QDQ/layout commuting."""
from __future__ import annotations

from merlin.frontends.linalg_mlir import parse_mlir_text
from mlir_oot.frontend.gemmini_friendly_quant import (
    _static_per_tensor_i8_layout_chain,
    lower_contraction_int8,
)


_RESHAPED_MATMUL = r"""
builtin.module { func.func @forward(
    %aq: tensor<2x3xi8>, %as: tensor<f32>,
    %wq: tensor<3x4xi8>, %ws: tensor<f32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : i64
  %zp = tensor.splat %c0 : tensor<i64>
  %ad = "quant_ext.dequantize_per_tensor"(%aq, %as, %zp)
    <{quant_min = -128 : i64, quant_max = 127 : i64}> :
    (tensor<2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<2x3xf32>
  %ac = tensor.collapse_shape %ad [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %ar = tensor.expand_shape %ac [[0, 1]] output_shape [2, 3] :
    tensor<6xf32> into tensor<2x3xf32>
  %wd = "quant_ext.dequantize_per_tensor"(%wq, %ws, %zp)
    <{quant_min = -127 : i64, quant_max = 127 : i64}> :
    (tensor<3x4xi8>, tensor<f32>, tensor<i64>) -> tensor<3x4xf32>
  %wc = tensor.collapse_shape %wd [[0, 1]] : tensor<3x4xf32> into tensor<12xf32>
  %wr = tensor.expand_shape %wc [[0, 1]] output_shape [3, 4] :
    tensor<12xf32> into tensor<3x4xf32>
  %e = tensor.empty() : tensor<2x4xf32>
  %z = arith.constant 0.0 : f32
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
  %y = linalg.matmul ins(%ar, %wr : tensor<2x3xf32>, tensor<3x4xf32>)
    outs(%f : tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %y : tensor<2x4xf32>
} }
"""


_PADDED_GATHER = r"""
builtin.module { func.func @forward(%q: tensor<1x1x2x2xi8>, %s: tensor<f32>)
    -> tensor<1x1x2x2xf32> {
  %c0 = arith.constant 0 : i64
  %zp = tensor.splat %c0 : tensor<i64>
  %d = "quant_ext.dequantize_per_tensor"(%q, %s, %zp)
    <{quant_min = -128 : i64, quant_max = 127 : i64}> :
    (tensor<1x1x2x2xi8>, tensor<f32>, tensor<i64>) -> tensor<1x1x2x2xf32>
  %zf = arith.constant 0.0 : f32
  %base = tensor.splat %zf : tensor<1x1x4x4xf32>
  %pad = "tensor.insert_slice"(%d, %base) <{
    static_offsets = array<i64: 0, 0, 1, 1>,
    static_sizes = array<i64: 1, 1, 2, 2>,
    static_strides = array<i64: 1, 1, 1, 1>,
    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> :
    (tensor<1x1x2x2xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  %e = tensor.empty() : tensor<1x1x2x2xf32>
  %g = linalg.generic {indexing_maps = [
      affine_map<(d0,d1,d2,d3)->(d0,d1,d2+1,d3+1)>,
      affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel","parallel","parallel","parallel"]}
    ins(%pad : tensor<1x1x4x4xf32>) outs(%e : tensor<1x1x2x2xf32>) {
    ^bb0(%x: f32, %o: f32): linalg.yield %x : f32
  } -> tensor<1x1x2x2xf32>
  func.return %g : tensor<1x1x2x2xf32>
} }
"""


def test_both_calibrated_operands_survive_reshape_as_i8() -> None:
    module = parse_mlir_text(_RESHAPED_MATMUL)
    report: dict = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    module.verify()

    assert report == {
        "static_per_tensor_layout_reused": 2,
        "prequant_gather_erased_ops": 6,
    }
    contractions = [
        op for op in module.walk()
        if op.name == "linalg.generic" and len(op.inputs) == 2
        and str(op.results[0].type).endswith("xi32>")
    ]
    assert len(contractions) == 1
    assert [str(value.type) for value in contractions[0].inputs] == [
        "tensor<2x3xi8>", "tensor<3x4xi8>"]
    assert not any(
        getattr(getattr(op, "op_name", None), "data", "").startswith(
            "quant_ext.dequantize")
        for op in module.walk())
    assert not any(op.name == "math.absf" for op in module.walk())


def test_symmetric_zero_pad_and_pure_gather_commute_exactly() -> None:
    module = parse_mlir_text(_PADDED_GATHER)
    function = next(op for op in module.walk() if op.name == "func.func")
    returned = next(op for op in function.walk() if op.name == "func.return").operands[0]
    result = _static_per_tensor_i8_layout_chain(returned)
    assert result is not None
    value, scale, operations, _dead = result

    assert str(value.type) == "tensor<1x1x2x2xi8>"
    assert str(scale.type) == "tensor<f32>"
    assert [op.name for op in operations] == [
        "arith.constant", "tensor.splat", "tensor.insert_slice",
        "tensor.empty", "linalg.generic",
    ]
    assert str(operations[2].results[0].type) == "tensor<1x1x4x4xi8>"
    assert str(operations[-1].results[0].type) == "tensor<1x1x2x2xi8>"


def test_nonzero_padding_refuses() -> None:
    module = parse_mlir_text(_PADDED_GATHER.replace(
        "%zf = arith.constant 0.0 : f32", "%zf = arith.constant 1.0 : f32"))
    returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert _static_per_tensor_i8_layout_chain(returned) is None
