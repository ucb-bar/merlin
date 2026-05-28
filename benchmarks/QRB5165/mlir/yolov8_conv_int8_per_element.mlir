// NCHW int8 conv DAG with **per-element** weights and bias — for testing
// the `dense_to_bytes` extraction path and the OIhw→HWIO byte permutation
// in `_permute_oihw_to_hwio`. Same DAG shape as
// `yolov8_stem_conv_int8.mlir` but smaller (OC=2, IC=1, Kh=Kw=2 → 8
// weights) and with distinct per-element values so the permutation can
// be validated element-by-element.
//
// OIhw weight layout (16 = OC*IC*Kh*Kw):
//   weight[0, 0, :, :] =  [[ 1,  2],
//                          [ 3,  4]]
//   weight[1, 0, :, :] =  [[ 5,  6],
//                          [ 7,  8]]
// Flat OIhw bytes: 1 2 3 4 5 6 7 8

module {
  func.func @yolov8_conv_int8_per_element(%input: tensor<1x1x4x4xi8>)
      -> tensor<1x2x3x3xf32> {
    %c0_i32 = arith.constant 0 : i32

    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_zp_i32 = arith.constant 0.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_output_scale = arith.constant 1.000000e-01 : f32

    // Per-channel f32 bias (length OC=2, distinct values).
    %bias_f32 = arith.constant dense<[1.000000e-01, 2.000000e-01]>
        : tensor<2xf32>

    // Per-element i8 weight, OIhw [2, 1, 2, 2] = 8 distinct bytes.
    %weight_i8 = arith.constant dense<[
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]]
      ]> : tensor<2x1x2x2xi8>

    // Bias quantize: f32 → i32
    %bias_i32_init = tensor.empty() : tensor<2xi32>
    %bias_i32 = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%bias_f32 : tensor<2xf32>)
        outs(%bias_i32_init : tensor<2xi32>) {
      ^bb0(%in: f32, %out: i32):
        %d  = arith.divf %in, %cst_bias_scale : f32
        %r  = math.roundeven %d : f32
        %z  = arith.addf %r, %cst_zp_i32 : f32
        %lo = arith.maximumf %z, %cst_min_i32 : f32
        %hi = arith.minimumf %lo, %cst_max_i32 : f32
        %q  = arith.fptosi %hi : f32 to i32
        linalg.yield %q : i32
    } -> tensor<2xi32>

    // Pad input — no padding for this fixture, but recognizer expects
    // tensor.pad in the chain so we keep it with all-zeros.
    %padded = tensor.pad %input low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : i8
      tensor.yield %z : i8
    } : tensor<1x1x4x4xi8> to tensor<1x1x4x4xi8>

    // Broadcast bias across N=1, H=3, W=3
    %acc_init = tensor.empty() : tensor<1x2x3x3xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<2xi32>)
        outs(%acc_init : tensor<1x2x3x3xi32>)
        dimensions = [0, 2, 3]

    // Quantized NCHW conv (stride 1, no padding, kernel 2x2)
    %conv_i32 = linalg.conv_2d_nchw_fchw_q
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<1> : vector<2xi64>}
        ins(%padded, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x1x4x4xi8>, tensor<2x1x2x2xi8>, i32, i32)
        outs(%broadcasted : tensor<1x2x3x3xi32>) -> tensor<1x2x3x3xi32>

    // Dequantize i32 → f32
    %out_init = tensor.empty() : tensor<1x2x3x3xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x2x3x3xi32>)
        outs(%out_init : tensor<1x2x3x3xf32>) {
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_output_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x2x3x3xf32>

    return %out : tensor<1x2x3x3xf32>
  }
}
