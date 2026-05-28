// Pattern-input probe — deterministic mixed-sign A and B (filled by the runner
// with -DMATMUL_INPUT_PATTERN=1) feed the standard 1x1x2048 linalg.matmul.
// Reference is computed in the runner via naive int8 matmul; gemmini result
// must match bit-for-bit. The previous all-ones fixture (matmul_1x1x2048_tensor.mlir)
// happens to use unit values everywhere, which can hide sign-extension /
// saturation bugs that only manifest for negative inputs (as in dronet's
// quantized conv-stack activation that feeds the FC heads).
func.func @matmul_1x1x2048_pattern(%A: tensor<1x2048xi8>, %B: tensor<2048x1xi8>) -> tensor<1x1xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<1x1xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<1x1xi32>) -> tensor<1x1xi32>
  %res = linalg.matmul ins(%A, %B : tensor<1x2048xi8>, tensor<2048x1xi8>)
                       outs(%fill : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %res : tensor<1x1xi32>
}
