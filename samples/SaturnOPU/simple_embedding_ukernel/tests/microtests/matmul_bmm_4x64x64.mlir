// Phase D11: isolate batched matmul (3D contraction).
// Mirrors vit d6 shape (Q·K^T per head): 4 attention heads, each 64×64.
// If 2D matmul works but this hangs, batch-dim handling in the encoding
// resolver or ukernel is the bug.
//
// Uses f32 inputs to match model_benchmark.c's default input type.
// RHS is a constant so the harness only needs one input tensor.
func.func @main(%lhs: tensor<4x64x32xf32>) -> tensor<4x64x64xf32> {
  %rhs = arith.constant dense<0.0625> : tensor<4x32x64xf32>
  %acc_init = tensor.empty() : tensor<4x64x64xf32>
  %c0 = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%c0 : f32) outs(%acc_init : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  %0 = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x64x32xf32>, tensor<4x32x64xf32>)
                           outs(%acc : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
  return %0 : tensor<4x64x64xf32>
}
