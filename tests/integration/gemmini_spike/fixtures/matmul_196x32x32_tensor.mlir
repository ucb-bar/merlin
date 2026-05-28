// Auto-generated probe fixture for dronet × Gemmini matmul 196x32x32.
// All-ones A and B → expected i32 output = 32 at every cell.
func.func @matmul_196x32x32(%A: tensor<196x32xi8>, %B: tensor<32x32xi8>) -> tensor<196x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<196x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<196x32xi32>) -> tensor<196x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<196x32xi8>, tensor<32x32xi8>)
                       outs(%fill : tensor<196x32xi32>) -> tensor<196x32xi32>
  return %res : tensor<196x32xi32>
}
