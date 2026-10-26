builtin.module {
  func.func @k() -> tensor<2xi8> {
    %c = arith.constant dense_resource<blob1> : tensor<2xi8>
    func.return %c : tensor<2xi8>
  }
}