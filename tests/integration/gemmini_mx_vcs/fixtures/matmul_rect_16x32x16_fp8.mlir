// 16x16x32 rectangular matmul (A=16x32, B=32x16). Same input pattern.
// Per-cell result: 32 × 2.0 × 2.0 = 128.0 = BF16 0x4300.

func.func @mlp_3layer(%input: tensor<16x32xi8>) -> tensor<16x16xi32>
    attributes {iree.abi.async = true} {
  %c0_i32 = arith.constant 0 : i32
  %w = arith.constant dense<64> : tensor<32x16xi8>
  %init = tensor.empty() : tensor<16x16xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%init : tensor<16x16xi32>) -> tensor<16x16xi32>
  %r = linalg.matmul ins(%input, %w : tensor<16x32xi8>, tensor<32x16xi8>)
      outs(%fill : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %r : tensor<16x16xi32>
}
