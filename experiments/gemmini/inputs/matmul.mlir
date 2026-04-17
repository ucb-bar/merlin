module {
  func.func @matmul(%A: memref<64x64xf16>, %B: memref<64x64xf16>, %C: memref<64x64xf32>) {
    linalg.matmul ins(%A, %B : memref<64x64xf16>, memref<64x64xf16>)
                 outs(%C : memref<64x64xf32>)
    return
  }
}
