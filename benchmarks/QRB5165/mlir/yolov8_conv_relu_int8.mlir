// NCHW int8 conv + dequant + ReLU — the simplest fusion shape the
// recognizer needs to match for HTA's `fold_relu_activation_into_conv`
// optimizer to collapse Conv2D and ElementWiseNeuron(Relu) into one HVX
// pass (closes the Conv+Relu portion of PR-D #95).
//
// Same conv shape as `yolov8_stem_conv_int8.mlir` (input 1×3×8×8 i8,
// weight 16×3×3×3 i8, stride 2, pad 1, output NCHW i8) but with a
// trailing ReLU `linalg.generic` (max-with-zero) on the f32 dequant
// result. yolov8 itself uses SiLU; this fixture isolates the simpler
// ReLU shape so the recognizer's optional-activation logic can be
// validated independently.

module {
  func.func @yolov8_conv_relu_int8(%input: tensor<1x3x8x8xi8>)
      -> tensor<1x16x4x4xf32> {
    %c0_i32 = arith.constant 0 : i32

    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_zp_i32 = arith.constant 0.000000e+00 : f32
    %cst_zero_f32 = arith.constant 0.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_output_scale = arith.constant 1.000000e-01 : f32

    %bias_f32 = arith.constant dense<0.000000e+00> : tensor<16xf32>
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

    // 2. Pad input
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : i8
      tensor.yield %z : i8
    } : tensor<1x3x8x8xi8> to tensor<1x3x10x10xi8>

    // 3. Broadcast bias
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

    // 5. Dequantize i32 -> f32
    %deq_init = tensor.empty() : tensor<1x16x4x4xf32>
    %deq = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x16x4x4xi32>)
        outs(%deq_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_output_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    // 6. ReLU: maximumf with 0 (the fusion target)
    %relu_init = tensor.empty() : tensor<1x16x4x4xf32>
    %relu = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%deq : tensor<1x16x4x4xf32>)
        outs(%relu_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: f32, %y: f32):
        %r = arith.maximumf %in, %cst_zero_f32 : f32
        linalg.yield %r : f32
    } -> tensor<1x16x4x4xf32>

    return %relu : tensor<1x16x4x4xf32>
  }
}
