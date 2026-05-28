// Real-shape yolov8 NCHW int8 conv + SiLU.
//
// Mirrors the 7-generic chain that real IREE-emitted yolov8 IR
// produces around the SiLU activation:
//
//   1. conv_2d_nchw_fchw_q → i32              (the conv)
//   2. dequant1: i32 → f32  (mulf cst_203)    (initial f32)
//   3. quantize1: f32 → i8  (divf cst_204)    \
//   4. dequant2:  i8 → f32  (mulf cst_204)    /  requantize round-trip 1
//   5. sigmoid: negf, exp, addf 1, divf 1     (SiLU's sigmoid)
//   6. quantize2: f32 → i8  (divf cst_205)    \
//   7. dequant3:  i8 → f32  (mulf cst_205)    /  requantize round-trip 2
//   8. multiply: dequant2 * dequant3 → f32    (SiLU = x * sigmoid(x))
//
// The recognizer walks past the round-trips so its activation
// classifier sees the post-roundtrip f32 value, then escalates the
// detected sigmoid to SiLU when a multiply consuming both the
// post-conv-roundtrip value AND the post-sigmoid-roundtrip value is
// present.

module {
  func.func @yolov8_conv_silu_real_int8(%input: tensor<1x3x8x8xi8>)
      -> tensor<1x16x4x4xf32> {
    %c0_i32 = arith.constant 0 : i32

    %cst_min_i32 = arith.constant -2.147483648e+09 : f32
    %cst_max_i32 = arith.constant 2.147483647e+09 : f32
    %cst_min_i8  = arith.constant -1.280000e+02 : f32
    %cst_max_i8  = arith.constant  1.270000e+02 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    %cst_one_f32 = arith.constant 1.000000e+00 : f32
    %cst_bias_scale = arith.constant 1.250000e-03 : f32
    %cst_203 = arith.constant 1.000000e-01 : f32  // dequant1 scale
    %cst_204 = arith.constant 1.000000e-01 : f32  // requant-roundtrip-1 scale
    %cst_205 = arith.constant 1.000000e-01 : f32  // requant-roundtrip-2 scale

    %bias_f32 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %weight_i8 = arith.constant dense<1> : tensor<16x3x3x3xi8>

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

    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      %z = arith.constant 0 : i8
      tensor.yield %z : i8
    } : tensor<1x3x8x8xi8> to tensor<1x3x10x10xi8>

    %acc_init = tensor.empty() : tensor<1x16x4x4xi32>
    %broadcasted = linalg.broadcast
        ins(%bias_i32 : tensor<16xi32>)
        outs(%acc_init : tensor<1x16x4x4xi32>)
        dimensions = [0, 2, 3]

    // 1. Conv
    %conv_i32 = linalg.conv_2d_nchw_fchw_q
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<2> : vector<2xi64>}
        ins(%padded, %weight_i8, %c0_i32, %c0_i32 :
            tensor<1x3x10x10xi8>, tensor<16x3x3x3xi8>, i32, i32)
        outs(%broadcasted : tensor<1x16x4x4xi32>) -> tensor<1x16x4x4xi32>

    // 2. Dequantize (i32 → f32)
    %deq1_init = tensor.empty() : tensor<1x16x4x4xf32>
    %deq1 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%conv_i32 : tensor<1x16x4x4xi32>)
        outs(%deq1_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i32, %y: f32):
        %f = arith.sitofp %in : i32 to f32
        %s = arith.mulf %f, %cst_203 : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    // 3. Quantize (f32 → i8) — requantize round-trip 1
    %q1_init = tensor.empty() : tensor<1x16x4x4xi8>
    %q1 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%deq1 : tensor<1x16x4x4xf32>)
        outs(%q1_init : tensor<1x16x4x4xi8>) {
      ^bb0(%in: f32, %y: i8):
        %d  = arith.divf %in, %cst_204 : f32
        %r  = math.roundeven %d : f32
        %z  = arith.addf %r, %cst_zp : f32
        %lo = arith.maximumf %z, %cst_min_i8 : f32
        %hi = arith.minimumf %lo, %cst_max_i8 : f32
        %q  = arith.fptosi %hi : f32 to i8
        linalg.yield %q : i8
    } -> tensor<1x16x4x4xi8>

    // 4. Dequantize (i8 → f32) — completes round-trip 1; this is "x"
    %deq2_init = tensor.empty() : tensor<1x16x4x4xf32>
    %deq2 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%q1 : tensor<1x16x4x4xi8>)
        outs(%deq1_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_204 : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    // 5. Sigmoid
    %sig_init = tensor.empty() : tensor<1x16x4x4xf32>
    %sig = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%deq2 : tensor<1x16x4x4xf32>)
        outs(%sig_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: f32, %y: f32):
        %neg = arith.negf %in : f32
        %e   = math.exp %neg : f32
        %s   = arith.addf %e, %cst_one_f32 : f32
        %r   = arith.divf %cst_one_f32, %s : f32
        linalg.yield %r : f32
    } -> tensor<1x16x4x4xf32>

    // 6. Quantize sigmoid (f32 → i8) — requantize round-trip 2
    %q2_init = tensor.empty() : tensor<1x16x4x4xi8>
    %q2 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%sig : tensor<1x16x4x4xf32>)
        outs(%q2_init : tensor<1x16x4x4xi8>) {
      ^bb0(%in: f32, %y: i8):
        %d  = arith.divf %in, %cst_205 : f32
        %r  = math.roundeven %d : f32
        %z  = arith.addf %r, %cst_zp : f32
        %lo = arith.maximumf %z, %cst_min_i8 : f32
        %hi = arith.minimumf %lo, %cst_max_i8 : f32
        %q  = arith.fptosi %hi : f32 to i8
        linalg.yield %q : i8
    } -> tensor<1x16x4x4xi8>

    // 7. Dequantize sigmoid (i8 → f32) — completes round-trip 2
    %deq3_init = tensor.empty() : tensor<1x16x4x4xf32>
    %deq3 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%q2 : tensor<1x16x4x4xi8>)
        outs(%deq1_init : tensor<1x16x4x4xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_205 : f32
        linalg.yield %s : f32
    } -> tensor<1x16x4x4xf32>

    // 8. Multiply: x * sigmoid(x)
    %silu_init = tensor.empty() : tensor<1x16x4x4xf32>
    %silu = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%deq2, %deq3 : tensor<1x16x4x4xf32>, tensor<1x16x4x4xf32>)
        outs(%silu_init : tensor<1x16x4x4xf32>) {
      ^bb0(%a: f32, %b: f32, %y: f32):
        %m = arith.mulf %a, %b : f32
        linalg.yield %m : f32
    } -> tensor<1x16x4x4xf32>

    return %silu : tensor<1x16x4x4xf32>
  }
}
