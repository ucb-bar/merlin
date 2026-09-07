"""A constant quantized weight transpose must not become per-inference dequant/requant work."""
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8


IR = r'''builtin.module {
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
}'''


def test_quantized_weight_is_transposed_as_i8_without_runtime_requantization():
    module = parse_mlir_text(IR)
    report = {}
    assert lower_contraction_int8(module, report_out=report) == 1
    assert report["transposed_quantized_weight_reused"] == 1

    transposes = [op for op in module.walk() if op.name == "linalg.transpose"
                  and str(op.results[0].type) == "tensor<8x4xi8>"]
    assert len(transposes) == 1
    assert str(transposes[0].operands[0].type) == "tensor<4x8xi8>"

    # The block-argument activation in this tiny fixture still needs one dynamic
    # quantizer. A second abs/round/fptosi chain would be the forbidden weight
    # dequantize + requantize path.
    weight_path_names = [op.name for op in module.walk()]
    assert weight_path_names.count("math.absf") == 1
    assert weight_path_names.count("math.roundeven") == 1
    assert weight_path_names.count("arith.fptosi") == 1
