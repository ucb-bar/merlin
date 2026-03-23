func.func @main(%lhs: tensor<64x2048xi8>, %rhs: tensor<2048x64xi8>) -> tensor<64x64xi32> {
  %acc_init = tensor.empty() : tensor<64x64xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<64x64xi32>) -> tensor<64x64xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x2048xi8>, tensor<2048x64xi8>)
                     outs(%acc : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %0 : tensor<64x64xi32>
}
