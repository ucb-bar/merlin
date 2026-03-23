// conv1-bad-buddy.mlir - INTENTIONALLY WRONG to validate checksum testing
//
// This uses WRONG parameters (stride=1 instead of stride=2) to verify
// that our checksum comparison can detect failures.

module {
  func.func @conv1_bad(%input: memref<4x224x224x3xi8>,
                       %weights: memref<147x64xi8>,
                       %bias: memref<64xi32>,
                       %output: memref<12544x64xi8>)
      attributes { llvm.emit_c_interface } {
    // WRONG: Using stride=1 instead of correct stride=2
    // This should produce a completely different (wrong) output
    %c112 = arith.constant 112 : i64
    %c7 = arith.constant 7 : i64

    // INTENTIONAL BUG: stride=1 (should be 2)
    gemmini.tile_conv %input %weights %bias %output %c112 %c112 %c7
        {stride = 1, inputDilation = 1, kernelDilation = 1, padding = 3,
         act = 1, poolSize = 3, poolStride = 2, poolPadding = 1} :
        memref<4x224x224x3xi8> memref<147x64xi8> memref<64xi32> memref<12544x64xi8>
        i64 i64 i64
    return
  }
}
