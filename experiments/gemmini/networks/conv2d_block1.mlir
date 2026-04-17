module {
  func.func @conv2d_block1(
      %input: memref<1x32x32x32xf16>,   // [N,H,W,C_in]
      %filter: memref<3x3x32x64xf16>,   // [KH,KW,C_in,C_out]
      %output: memref<1x30x30x64xf32>   // [N,H_out,W_out,C_out]
  ) {
    // linalg conv2d in NHWC x HWCF
    linalg.conv_2d_nhwc_hwcf
      ins(%input, %filter : memref<1x32x32x32xf16>, memref<3x3x32x64xf16>)
      outs(%output : memref<1x30x30x64xf32>)
    return
  }
}
