func.func @matmul_196x32x288(%A: tensor<196x288xi8>, %B: tensor<288x32xi8>) -> tensor<196x32xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<196x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<196x32xi32>) -> tensor<196x32xi32>
  %res = linalg.matmul ins(%A, %B : tensor<196x288xi8>, tensor<288x32xi8>)
                       outs(%fill : tensor<196x32xi32>) -> tensor<196x32xi32>
  return %res : tensor<196x32xi32>
}
