func.func @main(%lhs: tensor<2048x2048xi8>, %rhs: tensor<2048x2048xi8>) -> tensor<2048x2048xi32> {
  %acc_init = tensor.empty() : tensor<2048x2048xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xi8>, tensor<2048x2048xi8>)
                      outs(%acc : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %out_init = tensor.empty() : tensor<2048x2048xi32>
  %t = linalg.transpose ins(%mm : tensor<2048x2048xi32>)
                        outs(%out_init : tensor<2048x2048xi32>)
                        permutation = [1, 0]
  return %t : tensor<2048x2048xi32>
}
