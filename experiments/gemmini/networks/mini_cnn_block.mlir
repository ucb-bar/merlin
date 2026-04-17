module {
  func.func @mini_cnn_block(
      %input: memref<1x3x32x32xf32>,      // NCHW input
      %w1:    memref<16x3x3x3xf32>,       // conv1 weights
      %w2:    memref<32x16x3x3xf32>,      // conv2 weights
      %out:   memref<1x32x26x26xf32>      // final output after conv2
  ) {
    %conv1 = memref.alloc() : memref<1x16x30x30xf32>
    %conv2 = memref.alloc() : memref<1x32x26x26xf32>

    // Conv 1: 3x3, stride 1, NCHW x FCHW
    linalg.conv_2d_nchw_fchw
      ins(%input, %w1
          : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
      outs(%conv1
          : memref<1x16x30x30xf32>)

    // Conv 2: 3x3, stride 1, NCHW x FCHW
    linalg.conv_2d_nchw_fchw
      ins(%conv1, %w2
          : memref<1x16x30x30xf32>, memref<32x16x3x3xf32>)
      outs(%conv2
          : memref<1x32x26x26xf32>)

    // Just copy conv2 -> out for now (no FC yet)
    linalg.copy
      ins(%conv2 : memref<1x32x26x26xf32>)
      outs(%out  : memref<1x32x26x26xf32>)

    memref.dealloc %conv1 : memref<1x16x30x30xf32>
    memref.dealloc %conv2 : memref<1x32x26x26xf32>
    return
  }
}
