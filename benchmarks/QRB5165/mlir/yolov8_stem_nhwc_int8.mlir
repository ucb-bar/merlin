// YOLOv8 nano stem conv — REAL shape, NHWC layout (matches Qualcomm's
// converter convention). Skips Transpose ops entirely so HTA accepts.
//
// Conv: input 1×320×320×3 i8 → conv 3×3 stride 2 with 16 filters →
//       output 1×160×160×16 i8.
// Weights HWIO: 3×3×3×16. Bias 16 channels sfixed_point_32.
//
// Same numerical operation as `yolov8_stem_conv_int8.mlir` but with
// the layout already pre-permuted on the IREE side. The recognizer
// matches `linalg.conv_2d_nhwc_hwcf_q` directly without inserting
// Transpose adapters; the resulting QNN graph is pure NHWC and HTA
// accepts it.

module {
  func.func @yolov8_stem_nhwc_int8(%input: tensor<1x320x320x3xi8>)
      -> tensor<1x160x160x16xf32> {
    %c0_i32 = arith.constant 0 : i32

    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_output_scale = arith.constant 1.000000e-01 : f32

    // 16-channel bias, pre-permuted weight HWIO.
    %bias_f32 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %weight_i8 = arith.constant dense<1> : tensor<3x3x3x16xi8>

    // Bias quantize f32 → i32 (per-channel)
    %bias_i32_init = tensor.empty() : tensor<16xi32>
    %bias_i32 = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%bias_f32 : tensor<16xf32>)
        outs(%bias_i32_init : tensor<16xi32>) {
      ^bb0(%in: f32, %out: i32):
        %d  = arith.divf %in, %cst_bias_scale : f32
        %r  = math.roundeven %d : f32
        %z  = arith.addf %r, %cst_zp : f32
        %lo = arith.maximumf %z, %cst_min_i32 : f32
        %hi = arith.minimumf %lo, %cst_max_i32 : f32
        %q  = arith.fptosi %hi : f32 to i32
        linalg.yield %q : i32
    } -> tensor<16xi32>

    // Pad input (NHWC: pad height + width dims = 1, 2)
    %padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : i8
      tensor.yield %z : i8
    } : tensor<1x320x320x3xi8> to tensor<1x322x322x3xi8>

    // Broadcast bias across H, W (NHWC dims 1, 2)
    %acc_init = tensor.empty() : tensor<1x160x160x16xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<16xi32>)
        outs(%acc_init : tensor<1x160x160x16xi32>)
        dimensions = [0, 1, 2]

    // NHWC quantized conv (kernel HWIO)
    %conv_i32 = linalg.conv_2d_nhwc_hwcf_q
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<2> : vector<2xi64>}
        ins(%padded, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x322x322x3xi8>, tensor<3x3x3x16xi8>, i32, i32)
        outs(%broadcasted : tensor<1x160x160x16xi32>) -> tensor<1x160x160x16xi32>

    // Dequant i32 → f32 (NHWC)
    %deq_init = tensor.empty() : tensor<1x160x160x16xf32>
    %deq = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x160x160x16xi32>)
        outs(%deq_init : tensor<1x160x160x16xf32>) {
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_output_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x160x160x16xf32>

    return %deq : tensor<1x160x160x16xf32>
  }
}
