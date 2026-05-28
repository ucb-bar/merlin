func.func @matmul_3136x32x27(%A: tensor<3136x27xi8>, %B: tensor<27x32xi8>) -> tensor<3136x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<3136x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<3136x32xi32>) -> tensor<3136x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<3136x27xi8>, tensor<27x32xi8>)
                       outs(%fill : tensor<3136x32xi32>) -> tensor<3136x32xi32>
  return %res : tensor<3136x32xi32>
}
