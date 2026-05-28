// Standalone NCHW int8 maxpool dispatch — yolov8 has 3 of these in
// the SPPF block (5x5 maxpool stride 1, padding 2).
//
// Pattern: dequant (i8 → f32) → tensor.pad → linalg.pooling_nchw_max
// → return f32. The matched func takes i8 input and returns f32; the
// QNN lowering inserts NCHW↔NHWC transposes around a PoolMax2d op
// followed by a Dequantize.

module {
  func.func @yolov8_maxpool_int8(%input: tensor<1x128x14x14xi8>)
      -> tensor<1x128x10x10xf32> {
    %cst_input_scale = arith.constant 1.000000e-01 : f32

    // Dequant i8 → f32
    %deq_init = tensor.empty() : tensor<1x128x14x14xf32>
    %deq = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%input : tensor<1x128x14x14xi8>)
        outs(%deq_init : tensor<1x128x14x14xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_input_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x128x14x14xf32>

    // Pool window (sentinel — never read at runtime; just shapes the kernel)
    %window = tensor.empty() : tensor<5x5xf32>

    // 5x5 max-pool, stride 1, no padding (input was already padded
    // upstream at the dispatch boundary)
    %pool_init = tensor.empty() : tensor<1x128x10x10xf32>
    %cst_min = arith.constant 0xFF800000 : f32  // -inf
    %fill = linalg.fill ins(%cst_min : f32)
              outs(%pool_init : tensor<1x128x10x10xf32>)
              -> tensor<1x128x10x10xf32>
    %pool = linalg.pooling_nchw_max
        {dilations = dense<1> : vector<2xi64>,
         strides = dense<1> : vector<2xi64>}
        ins(%deq, %window : tensor<1x128x14x14xf32>, tensor<5x5xf32>)
        outs(%fill : tensor<1x128x10x10xf32>) -> tensor<1x128x10x10xf32>

    return %pool : tensor<1x128x10x10xf32>
  }
}
