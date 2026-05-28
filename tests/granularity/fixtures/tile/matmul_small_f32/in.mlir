util.func public @main(
    %lhs: tensor<32x32xf32>,
    %rhs: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %r = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  util.return %r : tensor<32x32xf32>
}
