// One `quant_ext.dequantize_per_channel` op exactly as model2MLIR captured it (small_llama, int8
// recapture `small_llama_int8_consistent`, model.mlir, value %122). Only the op, its operand types and
// its attributes are kept; the surrounding function is the minimum needed to parse it. The compiler
// never constructs this op itself -- the frontend does -- so this is the evidence that the lowering's
// per-channel kind is reachable.
module {
  func.func @captured(%w: tensor<128x128xi8>, %s: tensor<128xf32>, %z: tensor<128xi32>) -> tensor<128x128xf32> {
    %r = "quant_ext.dequantize_per_channel"(%w, %s, %z) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    return %r : tensor<128x128xf32>
  }
}
