util.func public @main(
    %lhs: tensor<128x128xf32>,
    %rhs: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %r = linalg.matmul ins(%lhs, %rhs : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  util.return %r : tensor<128x128xf32>
}
