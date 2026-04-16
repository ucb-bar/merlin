// Phase D-gap-8: representative chunk from vit_small d7 (attention softmax).
// Lifted from .../saturn_opu_OPU_opu_bench_vit_small.q.int8/sources/
//   module_main_graph$async_dispatch_7.mlir:20-37
//
// Structure (mirrors d7):
//   dequant i8 → f32 on 4×64×64
//   linalg.softmax dimension(2)
//   quantize f32 → i8 with saturation clamp
func.func @main(%x: tensor<4x64x64xf32>) -> tensor<4x64x64xi8> {
  %cst   = arith.constant 0.0078125 : f32   // dequant scale
  %cst_0 = arith.constant 0.00390625 : f32  // quant step
  %cst_1 = arith.constant 0.0 : f32
  %cst_2 = arith.constant -128.0 : f32
  %cst_3 = arith.constant 127.0 : f32

  // Synthesize i8 input from f32 harness input.
  %in_init = tensor.empty() : tensor<4x64x64xi8>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%x : tensor<4x64x64xf32>) outs(%in_init : tensor<4x64x64xi8>) {
    ^bb0(%in: f32, %out: i8):
      %c = arith.fptosi %in : f32 to i8
      linalg.yield %c : i8
    } -> tensor<4x64x64xi8>

  %3 = tensor.empty() : tensor<4x64x64xi8>
  %4 = tensor.empty() : tensor<4x64x64xf32>

  // Dequantize i8 → f32 * scale.
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%2 : tensor<4x64x64xi8>) outs(%4 : tensor<4x64x64xf32>) {
    ^bb0(%in: i8, %out: f32):
      %8 = arith.sitofp %in : i8 to f32
      %9 = arith.mulf %8, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<4x64x64xf32>

  // Softmax along the last dim.
  %6 = linalg.softmax dimension(2) ins(%5 : tensor<4x64x64xf32>) outs(%4 : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>

  // Requant f32 → i8.
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%6 : tensor<4x64x64xf32>) outs(%3 : tensor<4x64x64xi8>) {
    ^bb0(%in: f32, %out: i8):
      %8 = arith.divf %in, %cst_0 : f32
      %9 = math.roundeven %8 : f32
      %10 = arith.addf %9, %cst_1 : f32
      %11 = arith.maximumf %10, %cst_2 : f32
      %12 = arith.minimumf %11, %cst_3 : f32
      %13 = arith.fptosi %12 : f32 to i8
      linalg.yield %13 : i8
    } -> tensor<4x64x64xi8>

  return %7 : tensor<4x64x64xi8>
}
