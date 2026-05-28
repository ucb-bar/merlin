// YOLOv8 nano stem conv island for HTA profiling.
//
// This is the channel-last, int8-output interior of the real stem dispatch:
//   input  1x322x322x3 i8   (CPU boundary owns CHW->HWC/pad)
//   weight 3x3x3x16 i8      (HWIO)
//   bias   16xi32
//   output 1x160x160x16 i8
//
// No Transpose/Dequantize/Quantize boundary ops are inside this island. That is
// exactly the form HTA can validate. Whole-dispatch yolov8 still needs a split
// pass to route the CHW/HWC boundary ops to CPU and this island to HTA.

module {
  func.func @yolov8_stem_nhwc_i8out_int8(%input: tensor<1x322x322x3xi8>)
      -> tensor<1x160x160x16xi8> {
    %c0_i32 = arith.constant 0 : i32
    %cst_min_i8 = arith.constant -1.280000e+02 : f32
    %cst_max_i8 = arith.constant 1.270000e+02 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    %cst_acc_scale = arith.constant 5.957487e-03 : f32
    %cst_out_scale = arith.constant 1.322218e+00 : f32

    %bias_i32 = arith.constant dense<0> : tensor<16xi32>
    %weight_i8 = arith.constant dense<1> : tensor<3x3x3x16xi8>

    %acc_init = tensor.empty() : tensor<1x160x160x16xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<16xi32>)
        outs(%acc_init : tensor<1x160x160x16xi32>)
        dimensions = [0, 1, 2]

    %conv_i32 = linalg.conv_2d_nhwc_hwcf_q
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<2> : vector<2xi64>}
        ins(%input, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x322x322x3xi8>, tensor<3x3x3x16xi8>, i32, i32)
        outs(%broadcasted : tensor<1x160x160x16xi32>) -> tensor<1x160x160x16xi32>

    %out_init = tensor.empty() : tensor<1x160x160x16xi8>
    %out = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x160x160x16xi32>)
        outs(%out_init : tensor<1x160x160x16xi8>) {
      ^bb0(%in: i32, %y: i8):
        %f = arith.sitofp %in : i32 to f32
        %scaled = arith.mulf %f, %cst_acc_scale : f32
        %requant = arith.divf %scaled, %cst_out_scale : f32
        %rounded = math.roundeven %requant : f32
        %with_zp = arith.addf %rounded, %cst_zp : f32
        %lo = arith.maximumf %with_zp, %cst_min_i8 : f32
        %hi = arith.minimumf %lo, %cst_max_i8 : f32
        %q = arith.fptosi %hi : f32 to i8
        linalg.yield %q : i8
    } -> tensor<1x160x160x16xi8>

    return %out : tensor<1x160x160x16xi8>
  }
}
