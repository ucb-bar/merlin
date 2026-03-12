module {
  func.func @mlp1(%a0: memref<64x832xi8>,
                  %w0: memref<832x2560xi8>,
                  %c0: memref<64x2560xi8>,
                  %d0: memref<64x2560xi32>,
                  %w1: memref<2560x2048xi8>,
                  %c1: memref<64x2048xi8>,
                  %d1: memref<64x2048xi32>,
                  %w2: memref<2048x1536xi8>,
                  %c2: memref<64x1536xi8>,
                  %d2: memref<64x1536xi32>,
                  %w3: memref<1536x1024xi8>,
                  %c3: memref<64x1024xi8>,
                  %d3: memref<64x1024xi32>,
                  %w4: memref<1024x512xi8>,
                  %c4: memref<64x512xi8>,
                  %d4: memref<64x512xi32>,
                  %w5: memref<512x64xi8>,
                  %c5: memref<64x64xi8>,
                  %d5: memref<64x64xi32>) attributes { llvm.emit_c_interface } {
    gemmini.tile_matmul %a0 %w0 %c0 %d0 {dataflow = 1, act = 1} :
      memref<64x832xi8> memref<832x2560xi8> memref<64x2560xi8> memref<64x2560xi32>
    gemmini.tile_matmul %c0 %w1 %c1 %d1 {dataflow = 1, act = 1} :
      memref<64x2560xi8> memref<2560x2048xi8> memref<64x2048xi8> memref<64x2048xi32>
    gemmini.tile_matmul %c1 %w2 %c2 %d2 {dataflow = 1, act = 1} :
      memref<64x2048xi8> memref<2048x1536xi8> memref<64x1536xi8> memref<64x1536xi32>
    gemmini.tile_matmul %c2 %w3 %c3 %d3 {dataflow = 1, act = 1} :
      memref<64x1536xi8> memref<1536x1024xi8> memref<64x1024xi8> memref<64x1024xi32>
    gemmini.tile_matmul %c3 %w4 %c4 %d4 {dataflow = 1, act = 1} :
      memref<64x1024xi8> memref<1024x512xi8> memref<64x512xi8> memref<64x512xi32>
    gemmini.tile_matmul %c4 %w5 %c5 %d5 {dataflow = 1, act = 1} :
      memref<64x512xi8> memref<512x64xi8> memref<64x64xi8> memref<64x64xi32>
    return
  }
}
