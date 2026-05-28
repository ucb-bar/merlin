func.func @matmul_16x32x32(%A: tensor<16x32xi8>, %B: tensor<32x32xi8>) -> tensor<16x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<16x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<16x32xi32>) -> tensor<16x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<16x32xi8>, tensor<32x32xi8>)
                       outs(%fill : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %res : tensor<16x32xi32>
}
