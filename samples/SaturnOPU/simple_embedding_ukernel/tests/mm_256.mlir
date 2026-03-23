func.func @main(%lhs: tensor<256x256xi8>, %rhs: tensor<256x256xi8>) -> tensor<256x256xi32> {
  %acc_init = tensor.empty() : tensor<256x256xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<256x256xi32>) -> tensor<256x256xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<256x256xi8>, tensor<256x256xi8>)
                     outs(%acc : tensor<256x256xi32>) -> tensor<256x256xi32>
  return %0 : tensor<256x256xi32>
}
