func.func @matmul_i8(%arg0: tensor<1024x1024xi8>, %arg1: tensor<1024x1024xi8>) -> tensor<1024x1024xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %2 = linalg.matmul
    ins(%arg0, %arg1 : tensor<1024x1024xi8>, tensor<1024x1024xi8>)
    outs(%1 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  return %2 : tensor<1024x1024xi32>
}
