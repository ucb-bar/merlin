// 32x16x16 mxGemmini matmul — I-only tiling test (I=2, J=1, K=1).
// Per-cell expected: 16 × 2.0 × 2.0 = 64.0 = BF16 0x4280, i32 = 64.

func.func @mlp_3layer(%input: tensor<32x16xi8>) -> tensor<32x16xi32>
    attributes {iree.abi.async = true} {
  %c0_i32 = arith.constant 0 : i32
  %w = arith.constant dense<64> : tensor<16x16xi8>
  %init = tensor.empty() : tensor<32x16xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%init : tensor<32x16xi32>) -> tensor<32x16xi32>
  %r = linalg.matmul ins(%input, %w : tensor<32x16xi8>, tensor<16x16xi8>)
      outs(%fill : tensor<32x16xi32>) -> tensor<32x16xi32>
  return %r : tensor<32x16xi32>
}
