util.func public @main(
    %lhs: tensor<128x128xi8>,
    %rhs: tensor<128x128xi8>
) -> tensor<128x128xi8> {
  %cst = arith.constant 0 : i8
  %0 = tensor.empty() : tensor<128x128xi8>
  %init = linalg.fill ins(%cst : i8) outs(%0 : tensor<128x128xi8>) -> tensor<128x128xi8>
  %r = linalg.matmul ins(%lhs, %rhs : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%init : tensor<128x128xi8>) -> tensor<128x128xi8>
  util.return %r : tensor<128x128xi8>
}
