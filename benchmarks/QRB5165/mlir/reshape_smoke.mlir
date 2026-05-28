// fp32 reshape: collapse a 4D NHWC tensor to 2D (flatten spatial+channel).
// QNN's Reshape derives the target shape from the output tensor's
// declared dimensions; element count must match.
//
// [1, 6, 6, 4] (144 elems) → [1, 144]

module {
  func.func @reshape_f32(%input: tensor<1x6x6x4xf32>) -> tensor<1x144xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]]
      : tensor<1x6x6x4xf32> into tensor<1x144xf32>
    return %out : tensor<1x144xf32>
  }
}
