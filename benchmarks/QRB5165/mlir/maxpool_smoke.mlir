// fp32 2D max-pool, 2x2 window, stride 2, no padding.
//
// Input  NHWC: [1, 8, 8, 4]
// Output NHWC: [1, 4, 4, 4]   (after stride-2 pooling)
//
// linalg's pooling op takes a "fake filter" tensor that defines the window
// shape but isn't actually read. We declare it as a tensor.empty with the
// same shape as the pool window (2×2).

module {
  func.func @maxpool_f32(%input: tensor<1x8x8x4xf32>)
      -> tensor<1x4x4x4xf32> {
    %cst_min = arith.constant 0xFF800000 : f32  // -inf for max-pool init
    %win = tensor.empty() : tensor<2x2xf32>

    %init = tensor.empty() : tensor<1x4x4x4xf32>
    %fill = linalg.fill ins(%cst_min : f32)
              outs(%init : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>

    %out = linalg.pooling_nhwc_max
              {dilations = dense<1> : tensor<2xi64>,
               strides = dense<2> : tensor<2xi64>}
              ins(%input, %win :
                  tensor<1x8x8x4xf32>, tensor<2x2xf32>)
              outs(%fill : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    return %out : tensor<1x4x4x4xf32>
  }
}
