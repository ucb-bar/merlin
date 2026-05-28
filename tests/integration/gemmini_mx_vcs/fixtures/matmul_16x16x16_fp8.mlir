// Tiny single-matmul fixture for mxGemmini debugging. Both A and B are
// constants set to 0x40 (= FP8 e4m3 value 2.0) so we control exactly
// what reaches MVIN. Expected per-cell result: 16 × 2.0 × 2.0 = 64.0
// = BF16 0x4280 — the same value the matmul_16x16_fp8_ref-baremetal
// reference produced.

func.func @mlp_3layer(%input: tensor<16x16xi8>) -> tensor<16x16xi32>
    attributes {iree.abi.async = true} {
  %c0_i32 = arith.constant 0 : i32

  // Weight forced to 0x40 = 2.0 in FP8 e4m3. Input %input comes from
  // the runner (which initialises it to 0x40 too via memset). We use
  // %input as the A operand so IREE can't constant-fold the matmul
  // away — the linalg.matmul must survive to the dispatch.
  %w = arith.constant dense<64> : tensor<16x16xi8>

  %init = tensor.empty() : tensor<16x16xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%init : tensor<16x16xi32>) -> tensor<16x16xi32>
  %r = linalg.matmul ins(%input, %w : tensor<16x16xi8>, tensor<16x16xi8>)
      outs(%fill : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %r : tensor<16x16xi32>
}
