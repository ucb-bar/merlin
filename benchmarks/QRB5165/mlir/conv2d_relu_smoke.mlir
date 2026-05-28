// Conv2D + ReLU smoke fixture for the MLIR→QNN graph emitter (PR-C0).
//
// Same shape and arithmetic as the hand-authored
// `benchmarks/QRB5165/kernels/abi/conv2d_relu_smoke_f32.qnn.cpp`, so the
// emitter's output can be cross-checked bytes-equal against the
// hand-authored kernel's GPU result.
//
// Layout: NHWC input, HWCF (kh, kw, in_ch, out_ch) weight — these are
// QNN's native layouts. All-ones weights, -1.0 bias, then ReLU. The
// expected output equals max(0, sum_over_3x3x3(input) - 1.0) for each of
// the 4 output channels (all channels identical because all-1.0 weights).

module {
  func.func @conv2d_relu(%input: tensor<1x8x8x3xf32>)
      -> tensor<1x6x6x4xf32> {
    %cst_zero = arith.constant 0.0 : f32
    %weight = arith.constant dense<1.0> : tensor<3x3x3x4xf32>
    %bias = arith.constant dense<-1.0> : tensor<4xf32>

    %init = tensor.empty() : tensor<1x6x6x4xf32>
    %fill = linalg.fill ins(%cst_zero : f32)
              outs(%init : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>

    %conv = linalg.conv_2d_nhwc_hwcf
              {dilations = dense<1> : tensor<2xi64>,
               strides = dense<1> : tensor<2xi64>}
              ins(%input, %weight :
                  tensor<1x8x8x3xf32>, tensor<3x3x3x4xf32>)
              outs(%fill : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>

    %biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv, %bias : tensor<1x6x6x4xf32>, tensor<4xf32>)
        outs(%init : tensor<1x6x6x4xf32>) {
      ^bb0(%a: f32, %b: f32, %o: f32):
        %s = arith.addf %a, %b : f32
        linalg.yield %s : f32
    } -> tensor<1x6x6x4xf32>

    %relu = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%biased : tensor<1x6x6x4xf32>)
        outs(%init : tensor<1x6x6x4xf32>) {
      ^bb0(%a: f32, %o: f32):
        %r = arith.maximumf %a, %cst_zero : f32
        linalg.yield %r : f32
    } -> tensor<1x6x6x4xf32>

    return %relu : tensor<1x6x6x4xf32>
  }
}
