func.func @main(%lhs: tensor<128x128xi8>, %rhs: tensor<128x128xi8>) -> tensor<128x128xi32> {
  %acc_init = tensor.empty() : tensor<128x128xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%acc : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}
