util.func public @main(
    %lhs: tensor<32x32xf32>,
    %rhs: tensor<32x32xf32>,
    %bias: tensor<32xf32>
) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
                      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %ba_init = tensor.empty() : tensor<32x32xf32>
  %ba = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<32x32xf32>, tensor<32xf32>)
    outs(%ba_init : tensor<32x32xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<32x32xf32>
  %relu_init = tensor.empty() : tensor<32x32xf32>
  %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%ba : tensor<32x32xf32>) outs(%relu_init : tensor<32x32xf32>) {
  ^bb0(%a: f32, %b: f32):
    %z = arith.constant 0.0 : f32
    %r = arith.maximumf %a, %z : f32
    linalg.yield %r : f32
  } -> tensor<32x32xf32>
  util.return %relu : tensor<32x32xf32>
}
