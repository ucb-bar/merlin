module {
  // NCHW input: [N, C, H, W] = [1, 3, 32, 32]
  // FCHW filter: [F, C, KH, KW] = [64, 3, 3, 3]
  // NCHW output: [1, 64, 30, 30] (no padding, stride 1)
  func.func @conv2d_block_nchw(
      %input:  memref<1x3x32x32xf32>,
      %filter: memref<64x3x3x3xf32>,
      %output: memref<1x64x30x30xf32>
  ) {
    linalg.conv_2d_nchw_fchw
      ins(%input, %filter :
          memref<1x3x32x32xf32>, memref<64x3x3x3xf32>)
      outs(%output :
          memref<1x64x30x30xf32>)
    return
  }
}
