// Tensor-domain 64x64x64 int8 matmul. Multiples of DIM=16 so libgemmini.so's
// default OS pipeline can run them without padding. Flows through the
// full IREE plugin pipeline (see matmul_8x8x8_tensor.mlir for the path).
func.func @matmul_64x64x64(%A: tensor<64x64xi8>, %B: tensor<64x64xi8>) -> tensor<64x64xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<64x64xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<64x64xi32>) -> tensor<64x64xi32>
  %res = linalg.matmul ins(%A, %B : tensor<64x64xi8>, tensor<64x64xi8>)
                       outs(%fill : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %res : tensor<64x64xi32>
}
