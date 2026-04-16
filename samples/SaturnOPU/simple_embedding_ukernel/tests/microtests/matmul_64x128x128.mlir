// Phase D12: matmul shape-matched to vit d9 (64 M × 128 N × 128 K).
// Compiled under both FLAGS_MODEL_OPU and FLAGS_MODEL_OPU_LLM to
// isolate the --iree-preprocessing-collapse-multi-n-contractions pass.
// If the OPU variant passes but OPU_LLM hangs, that pass is the
// tinyllama-specific culprit.
//
// f32 for harness compatibility (benchmark fills one f32 input tensor);
// constant RHS so only one input is needed.
func.func @main(%lhs: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %rhs = arith.constant dense<0.0625> : tensor<128x128xf32>
  %acc_init = tensor.empty() : tensor<64x128xf32>
  %c0 = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%c0 : f32) outs(%acc_init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x128xf32>)
                     outs(%acc : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
