// Standalone NCHW int8 reshape dispatch — yolov8 has 9 of these
// (collapse_shape + expand_shape combined).
//
// Pattern: i8 input → tensor.collapse_shape (or expand_shape) → return
// i8. The lowering uses QNN's Reshape op directly on i8 tensors;
// reshape is layout-invariant so no NHWC adapter is needed.

module {
  func.func @yolov8_reshape_int8(%input: tensor<1x32x80x80xi8>)
      -> tensor<1x204800xi8> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]]
        : tensor<1x32x80x80xi8> into tensor<1x204800xi8>
    return %out : tensor<1x204800xi8>
  }
}
