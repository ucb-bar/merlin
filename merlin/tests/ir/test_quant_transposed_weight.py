"""A constant quantized weight transpose must not become per-inference dequant/requant work."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.llvmlower.passes_xdsl import lower_bf16_matmul_f32acc
from merlin.llvmlower.quant_scope import captured_quantized_contraction
from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
from merlin.runtime.dispatch_runtime import _normalize_model_module
from merlin.xdsl_dialects._common import text as mlir_text

IR = r"""builtin.module {
  func.func @forward(%aq: tensor<1x8xi8>, %as: tensor<f32>, %az: tensor<i64>,
                     %wq: tensor<4x8xi8>, %ws: tensor<4xf32>, %wz: tensor<4xi64>)
      -> tensor<1x4xf32> {
    %a = "quant_ext.dequantize_per_tensor"(%aq, %as, %az)
      <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<1x8xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8xf32>
    %w = "quant_ext.dequantize_per_channel"(%wq, %ws, %wz)
      <{axis = 0 : i64, quant_min = -127 : i64, quant_max = 127 : i64}>
      : (tensor<4x8xi8>, tensor<4xf32>, tensor<4xi64>) -> tensor<4x8xf32>
    %e = tensor.empty() : tensor<8x4xf32>
    %wt = linalg.transpose ins(%w : tensor<4x8xf32>)
      outs(%e : tensor<8x4xf32>) permutation = [1, 0]
    %o = tensor.empty() : tensor<1x4xf32>
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%o : tensor<1x4xf32>) -> tensor<1x4xf32>
    %r = linalg.matmul ins(%a, %wt : tensor<1x8xf32>, tensor<8x4xf32>)
      outs(%f : tensor<1x4xf32>) -> tensor<1x4xf32>
    func.return %r : tensor<1x4xf32>
  }
}"""


def test_quantized_weight_is_transposed_as_i8_without_runtime_requantization():
    module = parse_mlir_text(IR)
    report = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    assert report["transposed_quantized_weight_reused"] == 1

    transposes = [
        op for op in module.walk() if op.name == "linalg.transpose" and str(op.results[0].type) == "tensor<8x4xi8>"
    ]
    assert len(transposes) == 1
    assert str(transposes[0].operands[0].type) == "tensor<4x8xi8>"

    # The block-argument activation in this tiny fixture still needs one dynamic
    # quantizer. A second abs/round/fptosi chain would be the forbidden weight
    # dequantize + requantize path.
    weight_path_names = [op.name for op in module.walk()]
    assert weight_path_names.count("math.absf") == 1
    assert weight_path_names.count("math.roundeven") == 1
    assert weight_path_names.count("arith.fptosi") == 1


def _per_tensor_ir(zero_point):
    ir = IR.replace("%ws: tensor<4xf32>, %wz: tensor<4xi64>", "%ws: tensor<f32>, %wz: tensor<i64>")
    ir = ir.replace(
        '    %w = "quant_ext.dequantize_per_channel"(%wq, %ws, %wz)',
        f"    %zpv = arith.constant {zero_point} : i64\n"
        "    %zpt = tensor.splat %zpv : tensor<i64>\n"
        '    %w = "quant_ext.dequantize_per_tensor"(%wq, %ws, %zpt)',
    )
    ir = ir.replace(
        "<{axis = 0 : i64, quant_min = -127 : i64, quant_max = 127 : i64}>",
        "<{quant_min = -127 : i64, quant_max = 127 : i64}>",
    )
    return ir.replace("(tensor<4x8xi8>, tensor<4xf32>, tensor<4xi64>)", "(tensor<4x8xi8>, tensor<f32>, tensor<i64>)")


def test_per_tensor_quantized_transpose_keeps_calibrated_bytes_and_scalar_scale():
    module = parse_mlir_text(_per_tensor_ir(0))
    report = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    assert report["transposed_quantized_weight_reused"] == 1
    assert sum(op.name == "math.roundeven" for op in module.walk()) == 1
    transposed = next(
        op for op in module.walk() if op.name == "linalg.transpose" and str(op.results[0].type) == "tensor<8x4xi8>"
    )
    assert transposed.inputs[0] is next(op for op in module.walk() if op.name == "func.func").body.block.args[3]
    # The generated requant generic must index its scalar weight scale with ().
    requant = [op for op in module.walk() if op.name == "linalg.generic"][-1]
    assert any(len(am.data.results) == 0 for am in requant.indexing_maps)


def test_nonzero_per_tensor_zero_point_is_not_reused():
    module = parse_mlir_text(_per_tensor_ir(3))
    report = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    assert not report.get("transposed_quantized_weight_reused", 0)
    assert sum(op.name == "math.roundeven" for op in module.walk()) == 2


@pytest.mark.parametrize("zero_point", [0, 3])
def test_static_activation_keeps_scale_through_flatten_and_expand(zero_point):
    source = _per_tensor_ir(0).replace("%aq: tensor<1x8xi8>", "%aq: tensor<1x1x8xi8>")
    source = source.replace(
        '    %a = "quant_ext.dequantize_per_tensor"(%aq, %as, %az)',
        f"    %azv = arith.constant {zero_point} : i64\n"
        "    %azt = tensor.splat %azv : tensor<i64>\n"
        '    %a3 = "quant_ext.dequantize_per_tensor"(%aq, %as, %azt)',
    )
    source = source.replace(
        "(tensor<1x8xi8>, tensor<f32>, tensor<i64>) -> tensor<1x8xf32>",
        "(tensor<1x1x8xi8>, tensor<f32>, tensor<i64>) -> tensor<1x1x8xf32>\n"
        "    %af = tensor.collapse_shape %a3 [[0, 1, 2]] : tensor<1x1x8xf32> into tensor<8xf32>\n"
        "    %a = tensor.expand_shape %af [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>",
    )
    module = parse_mlir_text(source)
    report = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    assert report.get("static_activation_reshape_reused", 0) == int(zero_point == 0)
    assert sum(op.name == "math.roundeven" for op in module.walk()) == int(zero_point != 0)
    module.verify()


def test_captured_bf16_dequant_contraction_reaches_integer_path():
    source = _per_tensor_ir(0)
    for shape in ("1x8", "4x8", "8x4", "1x4"):
        source = source.replace(f"tensor<{shape}xf32>", f"tensor<{shape}xbf16>")
    source = source.replace("%z = arith.constant 0.0 : f32", "%z = arith.constant 0.0 : bf16")
    source = source.replace("linalg.fill ins(%z : f32)", "linalg.fill ins(%z : bf16)")
    module = parse_mlir_text(source)
    assert lower_bf16_matmul_f32acc(module) == 1
    report = {}
    assert lower_contraction_int8(module, select=captured_quantized_contraction, report_out=report) == 1
    assert report["transposed_quantized_weight_reused"] == 1
    assert any(
        op.name == "linalg.generic" and str(op.results[0].type) == "tensor<1x4xi32>"
        for op in module.walk()
        if op.results
    )
    module.verify()


def test_captured_bf16_host_reference_matches_compiled_preparation(tmp_path):
    source = _per_tensor_ir(0)
    # A half-width dequantization rounds once from an exact binary64 product, which needs a bounded
    # zero point: a constant, not a run-time i64 argument.
    source = source.replace(
        '    %a = "quant_ext.dequantize_per_tensor"(%aq, %as, %az)',
        "    %azv = arith.constant 0 : i64\n"
        "    %azt = tensor.splat %azv : tensor<i64>\n"
        '    %a = "quant_ext.dequantize_per_tensor"(%aq, %as, %azt)',
    )
    for shape in ("1x8", "4x8", "8x4", "1x4"):
        source = source.replace(f"tensor<{shape}xf32>", f"tensor<{shape}xbf16>")
    source = source.replace("%z = arith.constant 0.0 : f32", "%z = arith.constant 0.0 : bf16")
    source = source.replace("linalg.fill ins(%z : f32)", "linalg.fill ins(%z : bf16)")
    model = tmp_path / "model.mlir"
    model.write_text(source)

    host = parse_mlir_text(source)
    _normalize_model_module(
        host,
        int8_compute=True,
        quant_passes=["contraction_int8"],
        quant_select=captured_quantized_contraction,
        prequant_gather=False,
    )
    compiled = _prepare_model_mlir(model, tmp_path, int8_compute=True, respect_captured_quantization_scope=True)
    assert mlir_text(host, generic=True) == mlir_text(parse_mlir_text(compiled.read_text()), generic=True)


_WIDEN = """    %{out} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel"]}}
        ins(%{src} : tensor<{shape}xbf16>) outs(%{empty} : tensor<{shape}xf32>) {{
    ^bb0(%x: bf16, %y: f32):
      %v = arith.extf %x : bf16 to f32
      linalg.yield %v : f32
    }} -> tensor<{shape}xf32>
"""


def _widened_ir():
    """The f32 matmul a frontend writes for a half-precision one: both captured operands are
    dequantized to bf16 (the weight then transposed), and an identity generic holding one
    ``arith.extf`` widens each in front of the contraction."""
    source = IR.replace("-> tensor<1x8xf32>\n    %w =", "-> tensor<1x8xbf16>\n    %w =")
    source = source.replace("-> tensor<4x8xf32>\n    %e =", "-> tensor<4x8xbf16>\n    %e =")
    source = source.replace(
        "    %e = tensor.empty() : tensor<8x4xf32>\n"
        "    %wt = linalg.transpose ins(%w : tensor<4x8xf32>)\n"
        "      outs(%e : tensor<8x4xf32>) permutation = [1, 0]\n",
        "    %e = tensor.empty() : tensor<8x4xbf16>\n"
        "    %wt = linalg.transpose ins(%w : tensor<4x8xbf16>)\n"
        "      outs(%e : tensor<8x4xbf16>) permutation = [1, 0]\n"
        "    %ea = tensor.empty() : tensor<1x8xf32>\n"
        + _WIDEN.format(out="a32", src="a", empty="ea", shape="1x8")
        + "    %ew = tensor.empty() : tensor<8x4xf32>\n"
        + _WIDEN.format(out="wt32", src="wt", empty="ew", shape="8x4"),
    )
    widened = source.replace("ins(%a, %wt : tensor<1x8xf32>", "ins(%a32, %wt32 : tensor<1x8xf32>")
    assert widened != IR and "%a32, %wt32" in widened
    return widened


def test_captured_operands_behind_exact_widenings_reuse_their_bytes():
    """Each widening is value preserving, so the selector admits the f32 matmul and the integer
    lowering peels the widenings: the captured weight bytes and their static scales are reused, and
    only the block-argument activation is quantized dynamically, exactly as without the widenings."""
    module = parse_mlir_text(_widened_ir())
    matmul = next(op for op in module.walk() if op.name == "linalg.matmul")
    assert captured_quantized_contraction(matmul)
    report = {}
    assert lower_contraction_int8(module, select=captured_quantized_contraction, report_out=report) == 1
    assert report["transposed_quantized_weight_reused"] == 1
    assert sum(op.name == "math.roundeven" for op in module.walk()) == 1
    transposed = next(
        op for op in module.walk() if op.name == "linalg.transpose" and str(op.results[0].type) == "tensor<8x4xi8>"
    )
    assert str(transposed.operands[0].type) == "tensor<4x8xi8>"
    module.verify()
