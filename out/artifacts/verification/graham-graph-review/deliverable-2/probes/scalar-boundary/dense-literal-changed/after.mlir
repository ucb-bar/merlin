builtin.module {
  func.func @k() -> tensor<2xi8> {
    %c = arith.constant dense<[1, 3]> : tensor<2xi8>
    func.return %c : tensor<2xi8>
  }
}