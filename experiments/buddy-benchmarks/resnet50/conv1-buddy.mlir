// conv1-buddy.mlir - Buddy MLIR for ResNet50 conv_1 layer
//
// Conv1 params: 7x7 conv, stride=2, padding=3, with 3x3 maxpool
// Input:  4 x 224 x 224 x 3   (batch x height x width x channels)
// Weights: 147 x 64  (patch_size=7*7*3 x out_channels)
// Bias: 64
// Output: 12544 x 64  (batch*pool_out_row*pool_out_col x out_channels)
//         = 4*56*56 x 64

module {
  func.func @conv1(%input: memref<4x224x224x3xi8>,
                   %weights: memref<147x64xi8>,
                   %bias: memref<64xi32>,
                   %output: memref<12544x64xi8>)
      attributes { llvm.emit_c_interface } {
    // out_row_dim and out_col_dim are BEFORE pooling
    %c112 = arith.constant 112 : i64
    %c7 = arith.constant 7 : i64

    // gemmini.tile_conv: input weights bias output outRowDim outColDim kernelDim
    // Attributes: stride, padding, act (1=ReLU), poolSize, poolStride, poolPadding
    // scale = 1.0 / (1 << 8) = 0.00390625 (from conv_1_params.output_scale)
    gemmini.tile_conv %input %weights %bias %output %c112 %c112 %c7
        {stride = 2, inputDilation = 1, kernelDilation = 1, padding = 3,
         act = 1, poolSize = 3, poolStride = 2, poolPadding = 1,
         scale = 0.00390625 : f32} :
        memref<4x224x224x3xi8> memref<147x64xi8> memref<64xi32> memref<12544x64xi8>
        i64 i64 i64
    return
  }
}
