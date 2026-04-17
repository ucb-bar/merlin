module {
  func.func @softmax_matmul(%a: memref<31x30xi8>,
                            %b: memref<30x66xi8>,
                            %c: memref<31x66xi8>,
                            %d: memref<31x66xi32>) attributes { llvm.emit_c_interface } {
    gemmini.tile_matmul %a %b %c %d {act = 4, bertScale = 0.05:f32, dataflow = 1} :
      memref<31x30xi8> memref<30x66xi8> memref<31x66xi8> memref<31x66xi32>
    return
  }
}
