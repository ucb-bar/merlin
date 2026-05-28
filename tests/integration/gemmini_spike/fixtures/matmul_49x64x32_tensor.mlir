func.func @matmul_49x64x32(%A: tensor<49x32xi8>, %B: tensor<32x64xi8>) -> tensor<49x64xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<49x64xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<49x64xi32>) -> tensor<49x64xi32>
  %res = linalg.matmul ins(%A, %B : tensor<49x32xi8>, tensor<32x64xi8>)
                       outs(%fill : tensor<49x64xi32>) -> tensor<49x64xi32>
  return %res : tensor<49x64xi32>
}
