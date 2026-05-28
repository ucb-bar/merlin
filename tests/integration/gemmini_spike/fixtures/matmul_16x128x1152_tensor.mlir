func.func @matmul_16x128x1152(%A: tensor<16x1152xi8>, %B: tensor<1152x128xi8>) -> tensor<16x128xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<16x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<16x128xi32>) -> tensor<16x128xi32>
  %res = linalg.matmul ins(%A, %B : tensor<16x1152xi8>, tensor<1152x128xi8>)
                       outs(%fill : tensor<16x128xi32>) -> tensor<16x128xi32>
  return %res : tensor<16x128xi32>
}
