func.func @matmul_8x32x16(%A: tensor<8x16xi8>, %B: tensor<16x32xi8>) -> tensor<8x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<8x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<8x32xi32>) -> tensor<8x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<8x16xi8>, tensor<16x32xi8>)
                       outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %res : tensor<8x32xi32>
}
