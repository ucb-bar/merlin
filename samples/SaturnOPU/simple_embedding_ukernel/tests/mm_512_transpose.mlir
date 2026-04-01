func.func @main(%lhs: tensor<512x512xi8>, %rhs: tensor<512x512xi8>) -> tensor<512x512xi32> {
  %acc_init = tensor.empty() : tensor<512x512xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<512x512xi32>) -> tensor<512x512xi32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<512x512xi8>, tensor<512x512xi8>)
                      outs(%acc : tensor<512x512xi32>) -> tensor<512x512xi32>
  %out_init = tensor.empty() : tensor<512x512xi32>
  %t = linalg.transpose ins(%mm : tensor<512x512xi32>)
                        outs(%out_init : tensor<512x512xi32>)
                        permutation = [1, 0]
  return %t : tensor<512x512xi32>
}
