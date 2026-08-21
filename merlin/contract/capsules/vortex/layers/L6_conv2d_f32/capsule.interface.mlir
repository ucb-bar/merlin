// L6_conv2d_f32: conv2d NHWC/HWCF, f32, unit stride, no padding.
// The NAMED linalg op, which is what a float torch export lowers to — so the backend meets conv in
// both the named and the generic form across the corpus and cannot rely on one spelling.
module attributes {merlin.capsule = "L6_conv2d_f32"} {
  func.func @forward(%IFM: tensor<1x8x8x4xf32> {merlin.role = "input"},
                     %W: tensor<3x3x4x8xf32> {merlin.role = "weight"}) -> tensor<1x6x6x8xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<1x6x6x8xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
    %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                   strides = dense<1> : tensor<2xi64>}
         ins(%IFM, %W : tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) outs(%init : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
    func.return %0 : tensor<1x6x6x8xf32>
  }
}
