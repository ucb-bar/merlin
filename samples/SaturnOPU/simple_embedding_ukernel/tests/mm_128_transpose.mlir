func.func @main(%lhs: tensor<128x128xi8>, %rhs: tensor<128x128xi8>) -> tensor<128x128xi32> {
  %acc_init = tensor.empty() : tensor<128x128xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<128x128xi8>, tensor<128x128xi8>)
                      outs(%acc : tensor<128x128xi32>) -> tensor<128x128xi32>
  %out_init = tensor.empty() : tensor<128x128xi32>
  %t = linalg.transpose ins(%mm : tensor<128x128xi32>)
                        outs(%out_init : tensor<128x128xi32>)
                        permutation = [1, 0]
  return %t : tensor<128x128xi32>
}
