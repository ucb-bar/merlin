// Match for the dronet stem conv: 1×224×224×1 → 1×112×112×32, kernel 5×5,
// stride 2, padding 2 (NHWC×HWCF as IREE's linalg.conv_2d_nhwc_hwcf
// expresses it).
^bb0(%input: tensor<1x224x224x1xf32>, %weight: tensor<5x5x1x32xf32>, %bias: tensor<32xf32>):
%cst_zero = arith.constant 0.0 : f32
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x112x112x32xf32>
%fill = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
%conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>,
     strides = dense<2> : tensor<2xi64>}
    ins(%input, %weight : tensor<1x224x224x1xf32>, tensor<5x5x1x32xf32>)
    outs(%fill : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv, %bias : tensor<1x112x112x32xf32>, tensor<32xf32>)
    outs(%init : tensor<1x112x112x32xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
} -> tensor<1x112x112x32xf32>
