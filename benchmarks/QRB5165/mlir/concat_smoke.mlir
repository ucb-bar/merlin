// fp32 channel-wise concat of two NHWC tensors. yolov8's neck has 17
// of these (mostly along the channel dim, dim=3 in NHWC).
//
// Shape: two [1,6,6,4] inputs → [1,6,6,8] output.

module {
  func.func @concat_f32(%a: tensor<1x6x6x4xf32>, %b: tensor<1x6x6x4xf32>)
      -> tensor<1x6x6x8xf32> {
    %out = tensor.concat dim(3) %a, %b :
      (tensor<1x6x6x4xf32>, tensor<1x6x6x4xf32>) -> tensor<1x6x6x8xf32>
    return %out : tensor<1x6x6x8xf32>
  }
}
