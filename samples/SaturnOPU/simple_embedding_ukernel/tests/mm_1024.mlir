func.func @main(%lhs: tensor<1024x1024xi8>, %rhs: tensor<1024x1024xi8>) -> tensor<1024x1024xi32> {
  %acc_init = tensor.empty() : tensor<1024x1024xi32>
  %c0 = arith.constant 0 : i32
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<1024x1024xi8>, tensor<1024x1024xi8>)
                     outs(%acc : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  return %0 : tensor<1024x1024xi32>
}
