module {
  func.func @mlp2(%a0: memref<64x832xi8>,
                  %w0: memref<832x832xi8>,
                  %c0: memref<64x832xi8>,
                  %d0: memref<64x832xi32>,
                  %w1: memref<832x64xi8>,
                  %c1: memref<64x64xi8>,
                  %d1: memref<64x64xi32>) attributes { llvm.emit_c_interface } {
    gemmini.tile_matmul %a0 %w0 %c0 %d0 {dataflow = 0, act = 1} :
      memref<64x832xi8> memref<832x832xi8> memref<64x832xi8> memref<64x832xi32>
    gemmini.tile_matmul %c0 %w1 %c1 %d1 {dataflow = 0, act = 1} :
      memref<64x832xi8> memref<832x64xi8> memref<64x64xi8> memref<64x64xi32>
    return
  }
}
