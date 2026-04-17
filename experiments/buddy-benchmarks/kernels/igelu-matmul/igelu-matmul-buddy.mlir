module {
  func.func @igelu_matmul(%a: memref<30x30xi8>,
                          %b: memref<30x30xi8>,
                          %c: memref<30x30xi8>,
                          %d: memref<30x30xi32>) attributes { llvm.emit_c_interface } {
    gemmini.tile_matmul %a %b %c %d {act = 3, bertScale = 0.8:f32, dataflow = 1} :
      memref<30x30xi8> memref<30x30xi8> memref<30x30xi8> memref<30x30xi32>
    return
  }
}
