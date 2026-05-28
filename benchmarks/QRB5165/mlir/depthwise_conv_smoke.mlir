// fp32 depthwise conv 3x3 stride 1, no padding, no bias.
//
// Input  NHWC: [1, 8, 8, 4]
// Weight HWC : [3, 3, 4]   (per-channel filters, all-ones)
// Output NHWC: [1, 6, 6, 4] (after VALID 3x3 conv on 8x8)
//
// Reference output[n,h,w,c] = sum_{kh,kw}(input[n, h+kh, w+kw, c]) for each
// of the 4 channels (each channel has its own 3x3 all-ones kernel).

module {
  func.func @depthwise_conv_f32(%input: tensor<1x8x8x4xf32>)
      -> tensor<1x6x6x4xf32> {
    %cst_zero = arith.constant 0.0 : f32
    %weight = arith.constant dense<1.0> : tensor<3x3x4xf32>

    %init = tensor.empty() : tensor<1x6x6x4xf32>
    %fill = linalg.fill ins(%cst_zero : f32)
              outs(%init : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>

    %out = linalg.depthwise_conv_2d_nhwc_hwc
              {dilations = dense<1> : tensor<2xi64>,
               strides = dense<1> : tensor<2xi64>}
              ins(%input, %weight :
                  tensor<1x8x8x4xf32>, tensor<3x3x4xf32>)
              outs(%fill : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>
    return %out : tensor<1x6x6x4xf32>
  }
}
