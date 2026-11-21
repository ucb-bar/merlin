builtin.module {
  func.func @flatten(%input: tensor<2x3xi32>) -> tensor<6xi32> {
    %flat = tensor.collapse_shape %input [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
    func.return %flat : tensor<6xi32>
  }
}
