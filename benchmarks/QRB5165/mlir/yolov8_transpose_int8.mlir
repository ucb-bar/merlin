// Standalone i8 transpose dispatch — yolov8 has 1 of these in the
// detection head. The recognizer reads the `permutation` attribute
// off `linalg.transpose` and emits a QNN Transpose node directly.

module {
  func.func @yolov8_transpose_int8(%input: tensor<1x32x80x80xi8>)
      -> tensor<1x80x80x32xi8> {
    %init = tensor.empty() : tensor<1x80x80x32xi8>
    %out = linalg.transpose ins(%input : tensor<1x32x80x80xi8>)
                            outs(%init : tensor<1x80x80x32xi8>)
                            permutation = [0, 2, 3, 1]
    return %out : tensor<1x80x80x32xi8>
  }
}
