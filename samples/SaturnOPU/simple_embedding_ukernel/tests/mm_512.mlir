func.func @main(%lhs: tensor<512x512xi8>, %rhs: tensor<512x512xi8>) -> tensor<512x512xi32> {
  %acc_init = tensor.empty() : tensor<512x512xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<512x512xi32>) -> tensor<512x512xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<512x512xi8>, tensor<512x512xi8>)
                     outs(%acc : tensor<512x512xi32>) -> tensor<512x512xi32>
  return %0 : tensor<512x512xi32>
}
