// Bias-path probe fixture — same shape as matmul_196x32x32_tensor.mlir but
// fills the output buffer with 1s instead of 0s before the matmul, so the
// linalg.matmul sees a non-zero D operand. After gemmini lowering the
// resulting gemmini.matmul has noBias=false (the D memref's shape contains
// no zero, so the `for (int64_t d : dMemRefType.getShape())` check in
// tiledMatmulOuter doesn't trip), which routes through the BIAS path.
//
// With A=B=all-ones (loaded by the runner) and D=all-ones (from this
// fill), the expected i32 output per cell is K + 1 = 32 + 1 = 33.
//
// Used by tools/spike-hetero to find the bias-path Gemmini bug that
// dronet hits but matmul_196x32x32_tensor.mlir (noBias) does not.
func.func @matmul_196x32x32_bias(%A: tensor<196x32xi8>, %B: tensor<32x32xi8>) -> tensor<196x32xi32> {
  %c1 = arith.constant 1 : i32
  %init = tensor.empty() : tensor<196x32xi32>
  %fill = linalg.fill ins(%c1 : i32) outs(%init : tensor<196x32xi32>) -> tensor<196x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<196x32xi8>, tensor<32x32xi8>)
                       outs(%fill : tensor<196x32xi32>) -> tensor<196x32xi32>
  return %res : tensor<196x32xi32>
}
