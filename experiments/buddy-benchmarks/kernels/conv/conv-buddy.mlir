module {
  func.func @conv(%input: memref<2x17x17x18xi8>,
                  %weights: memref<162x19xi8>,
                  %bias: memref<19xi32>,
                  %output: memref<162x19xi8>) attributes { llvm.emit_c_interface } {
    %c9 = arith.constant 9 : i64
    %c3 = arith.constant 3 : i64
    gemmini.tile_conv %input %weights %bias %output %c9 %c9 %c3
        {stride = 2, inputDilation = 1, kernelDilation = 1, padding = 1,
         act = 0} :
        memref<2x17x17x18xi8> memref<162x19xi8> memref<19xi32> memref<162x19xi8>
        i64 i64 i64
    return
  }
}
