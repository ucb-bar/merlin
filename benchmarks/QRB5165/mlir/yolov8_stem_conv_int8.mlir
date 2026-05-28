// YOLOv8 stem-conv DAG, real shape, real ops — the input to the
// `linalg.conv_2d_nchw_fchw_q` recognizer (#102, Phase 2.1).
//
// This mirrors a single conv slice from
//   build/compiled_models/yolov8_nano/qrb5165/phases/yolov8n_q_int8.1.input.mlir
// with deliberately small dims so unit tests stay fast. The op chain is:
//
//   1. Bias-quantize generic       : tensor<Cxf32>     -> tensor<Cxi32>
//        body = divf scale -> roundeven -> addf zp -> clamp -> fptosi
//   2. Pad input                   : tensor.pad     (zero pad H,W)
//   3. Empty + broadcast bias      : linalg.broadcast dims=[0,2,3]
//   4. Quantized conv NCHW×FCHW    : linalg.conv_2d_nchw_fchw_q
//        ins:  (padded i8, weight i8, input_zp i32, weight_zp i32)
//        outs: (broadcasted bias i32)
//   5. Dequantize generic          : tensor<NxCxHxWxi32> -> tensor<NxCxHxWxf32>
//        body = sitofp -> mulf output_scale
//
// Shapes (small): input 1×3×8×8, weight 16×3×3×3 (OC×IC×Kh×Kw),
// bias 16, stride 2, pad 1, output 1×16×4×4.
//
// q-params: input_scale=0.05 input_zp=0; weight_scale=0.025 weight_zp=0;
// bias_scale = 0.05*0.025 = 0.00125; output_scale=0.10.

module {
  func.func @yolov8_stem_conv_int8(%input: tensor<1x3x8x8xi8>)
      -> tensor<1x16x4x4xf32> {
    %c0_i32 = arith.constant 0 : i32

    // Quantize-bound clamping range (i32)
    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_zp_i32 = arith.constant 0.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_output_scale = arith.constant 1.000000e-01 : f32

    // f32 bias (16 channels, all zeros for simplicity)
    %bias_f32 = arith.constant dense<0.000000e+00> : tensor<16xf32>

    // Quantized weight (NCHW = OIHW, all-ones)
    %weight_i8 = arith.constant dense<1> : tensor<16x3x3x3xi8>

    // 1. Bias-quantize: f32 -> i32
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
        %z  = arith.addf %r, %cst_zp_i32 : f32
        %lo = arith.maximumf %z, %cst_min_i32 : f32
        %hi = arith.minimumf %lo, %cst_max_i32 : f32
        %q  = arith.fptosi %hi : f32 to i32
        linalg.yield %q : i32
    } -> tensor<16xi32>

    // 2. Pad input with 1 on H and W (NCHW means dims 2 and 3)
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : i8
      tensor.yield %z : i8
    } : tensor<1x3x8x8xi8> to tensor<1x3x10x10xi8>

    // 3. Broadcast bias across N=1, H=4, W=4
    %acc_init = tensor.empty() : tensor<1x16x4x4xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<16xi32>)
        outs(%acc_init : tensor<1x16x4x4xi32>)
        dimensions = [0, 2, 3]

    // 4. Quantized NCHW conv
    %conv_i32 = linalg.conv_2d_nchw_fchw_q
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<2> : vector<2xi64>}
        ins(%padded, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x3x10x10xi8>, tensor<16x3x3x3xi8>, i32, i32)
        outs(%broadcasted : tensor<1x16x4x4xi32>) -> tensor<1x16x4x4xi32>

    // 5. Dequantize i32 -> f32 (apply output_scale)
    %out_init = tensor.empty() : tensor<1x16x4x4xf32>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x16x4x4xi32>)
        outs(%out_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_output_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    return %out : tensor<1x16x4x4xf32>
  }
}
