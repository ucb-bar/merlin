util.func public @main(
    %lhs: tensor<32x32xi8>,
    %rhs: tensor<32x32xi8>
) -> tensor<32x32xi8> {
  %cst = arith.constant 0 : i8
  %0 = tensor.empty() : tensor<32x32xi8>
  %init = linalg.fill ins(%cst : i8) outs(%0 : tensor<32x32xi8>) -> tensor<32x32xi8>
  %r = linalg.matmul ins(%lhs, %rhs : tensor<32x32xi8>, tensor<32x32xi8>)
                     outs(%init : tensor<32x32xi8>) -> tensor<32x32xi8>
  util.return %r : tensor<32x32xi8>
}
