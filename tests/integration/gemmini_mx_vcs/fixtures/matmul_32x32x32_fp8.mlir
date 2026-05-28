// 32x32x32 mxGemmini matmul — exercises multi-tile compute (4 tiles of 16x16
// in each spatial dim). Same input pattern as the single-tile fixture so the
// expected per-cell result is 32 × 2.0 × 2.0 = 128.0 = BF16 0x4300.

func.func @mlp_3layer(%input: tensor<32x32xi8>) -> tensor<32x32xi32>
    attributes {iree.abi.async = true} {
  %c0_i32 = arith.constant 0 : i32
  %w = arith.constant dense<64> : tensor<32x32xi8>
  %init = tensor.empty() : tensor<32x32xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%init : tensor<32x32xi32>) -> tensor<32x32xi32>
  %r = linalg.matmul ins(%input, %w : tensor<32x32xi8>, tensor<32x32xi8>)
      outs(%fill : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %r : tensor<32x32xi32>
}
