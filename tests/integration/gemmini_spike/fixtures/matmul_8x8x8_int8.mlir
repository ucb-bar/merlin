// Linalg-level fixture exercised by `./merlin spike`.
//
// The exported function `matmul_8x8x8` is intentionally small enough that
// libgemmini.so's DIM=16 OS pipeline can run it as a single tile (with
// padding K/J=8). After bufferization (driven by the gemmini compile
// pipeline) it lowers through:
//   linalg.matmul -> gemmini.matmul -> gemmini.matmul_tile
//                 -> gemmini.tile_matmul -> gemmini.intr.* -> RoCC ELF.
//
// `iree.preserve_func_visibility` keeps the symbol exported for linking.

func.func @matmul_8x8x8(%A: memref<8x8xi8>, %B: memref<8x8xi8>, %C: memref<8x8xi32>)
    attributes {iree.preserve_func_visibility = true} {
  linalg.matmul ins(%A, %B : memref<8x8xi8>, memref<8x8xi8>)
                outs(%C : memref<8x8xi32>)
  return
}
