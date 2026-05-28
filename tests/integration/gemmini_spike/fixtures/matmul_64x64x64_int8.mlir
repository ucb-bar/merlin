// 64x64x64 int8 -> i32 matmul. Shapes are multiples of DIM=16 so libgemmini's
// default OS pipeline can run them without padding.
//
// Marked xfail in test_matmul_64x64x64.py until the linalg->memref
// bufferization gap is closed in the dialect; see the dev-blog log for
// status.

func.func @matmul_64x64x64(%A: memref<64x64xi8>, %B: memref<64x64xi8>, %C: memref<64x64xi32>)
    attributes {iree.preserve_func_visibility = true} {
  linalg.matmul ins(%A, %B : memref<64x64xi8>, memref<64x64xi8>)
                outs(%C : memref<64x64xi32>)
  return
}
