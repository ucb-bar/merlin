// Tensor-domain int8 matmul fixture exercised by `./merlin compile`.
//
// Unlike the memref-domain fixture in matmul_8x8x8_int8.mlir, this version is
// intended to flow through the full IREE plugin pipeline:
//   linalg.matmul (tensor) -> [post-global-opt: gemmini plugin] gemmini.matmul
//                          -> ISA-tier ops -> bufferization (handled by IREE)
//                          -> codegen -> RISC-V ELF inside .vmfb
//
// 8x8x8 matches libgemmini.so DIM=16's smallest viable tile (with K/N=8 padding).
func.func @matmul_8x8x8(%A: tensor<8x8xi8>, %B: tensor<8x8xi8>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<8x8xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<8x8xi32>) -> tensor<8x8xi32>
  %res = linalg.matmul ins(%A, %B : tensor<8x8xi8>, tensor<8x8xi8>)
                       outs(%fill : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %res : tensor<8x8xi32>
}
