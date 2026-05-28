// Linalg-DAG match pattern for fp32 Conv2D + Relu over the smoke shape.
//
// Matches the canonical IREE form for a 3x3 fp32 conv with NHWC input,
// HWIO filter, and bias add, followed by an arith.maximumf-against-zero
// (the linalg.generic body shape ReLU lowers to). The two ops appear as
// two separate linalg.generic regions before fusion.

^bb0(%input: tensor<1x8x8x3xf32>, %weight: tensor<3x3x3x4xf32>, %bias: tensor<4xf32>, %conv_init: tensor<1x6x6x4xf32>, %relu_init: tensor<1x6x6x4xf32>):
%conv = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1+d4, d2+d5, d6)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
  } ins(%input, %weight, %bias : tensor<1x8x8x3xf32>, tensor<3x3x3x4xf32>, tensor<4xf32>)
    outs(%conv_init : tensor<1x6x6x4xf32>) {
  ^bb0(%i: f32, %w: f32, %b: f32, %out: f32):
    %m = arith.mulf %i, %w : f32
    %a = arith.addf %out, %m : f32
    %ab = arith.addf %a, %b : f32
    linalg.yield %ab : f32
} -> tensor<1x6x6x4xf32>

%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv : tensor<1x6x6x4xf32>)
    outs(%relu_init : tensor<1x6x6x4xf32>) {
  ^bb0(%a: f32, %o: f32):
    %zero = arith.constant 0.0 : f32
    %r = arith.maximumf %a, %zero : f32
    linalg.yield %r : f32
} -> tensor<1x6x6x4xf32>
