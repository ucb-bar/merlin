// Phase D-gap-6: representative chunk extracted from vit_small d1 (LayerNorm).
// Full computation lifted from the compiled dispatch:
//   build/compiled_models/.../saturn_opu_OPU_opu_bench_vit_small.q.int8/
//     sources/module_main_graph$async_dispatch_1.mlir:27-71
//
// Structure (mirrors d1):
//   dequant i8 → f32
//   sum reduction → mean
//   centered = x - mean/N
//   var = sum((x-mean)^2)/N
//   rsqrt(var + eps)
//   normalize + scale/bias + requant → i8
//
// Input harness feeds f32, so we cast to i8 first to synthesize an
// equivalent starting point. Everything after the cast matches d1 exactly.
func.func @main(%x: tensor<64x128xf32>) -> tensor<64x128xi8> {
  %cst = arith.constant 7.8125E-3 : f32    // dequant scale (= 1/128)
  %cst_0 = arith.constant 128.0 : f32      // feature dim N (divisor)
  %cst_1 = arith.constant 1.0E-5 : f32     // eps
  %cst_2 = arith.constant 0.00781 : f32    // quant out scale
  %cst_3 = arith.constant 0.0 : f32        // zero point
  %cst_4 = arith.constant -128.0 : f32     // i8 min
  %cst_5 = arith.constant 127.0 : f32      // i8 max

  // Synthesize i8 input from f32 harness input.
  %i8_in = tensor.empty() : tensor<64x128xi8>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<64x128xf32>) outs(%i8_in : tensor<64x128xi8>) {
    ^bb0(%in: f32, %out: i8):
      %c = arith.fptosi %in : f32 to i8
      linalg.yield %c : i8
    } -> tensor<64x128xi8>

  %4 = arith.constant dense<0.1> : tensor<128xf32>  // LN weight/bias
  %5 = tensor.empty() : tensor<64x128xi8>
  %6 = tensor.empty() : tensor<64x128xf32>
  %7 = tensor.empty() : tensor<64xf32>

  // Dequantize: i8 → f32 * scale
  %8 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%3 : tensor<64x128xi8>) outs(%6 : tensor<64x128xf32>) {
    ^bb0(%in: i8, %out: f32):
      %14 = arith.sitofp %in : i8 to f32
      %15 = arith.mulf %14, %cst : f32
      linalg.yield %15 : f32
    } -> tensor<64x128xf32>

  %9 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>

  // Sum reduction along d1.
  %10 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%8 : tensor<64x128xf32>) outs(%9 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %in, %out : f32
      linalg.yield %14 : f32
    } -> tensor<64xf32>

  // Centered = x - mean.
  %11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%8, %10 : tensor<64x128xf32>, tensor<64xf32>) outs(%6 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %14 = arith.divf %in_6, %cst_0 : f32
      %15 = arith.subf %in, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x128xf32>

  // Variance = sum(centered^2).
  %12 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%11 : tensor<64x128xf32>) outs(%9 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.mulf %in, %in : f32
      %15 = arith.addf %14, %out : f32
      linalg.yield %15 : f32
    } -> tensor<64xf32>

  // Normalize + scale/bias + requant (rsqrt is here!).
  %13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%11, %12, %4 : tensor<64x128xf32>, tensor<64xf32>, tensor<128xf32>)
    outs(%5 : tensor<64x128xi8>) {
    ^bb0(%in: f32, %in_6: f32, %in_7: f32, %out: i8):
      %14 = arith.divf %in_6, %cst_0 : f32
      %15 = arith.addf %14, %cst_1 : f32
      %16 = math.rsqrt %15 : f32
      %17 = arith.mulf %in, %16 : f32
      %18 = arith.addf %17, %in_7 : f32
      %19 = arith.divf %18, %cst_2 : f32
      %20 = math.roundeven %19 : f32
      %21 = arith.addf %20, %cst_3 : f32
      %22 = arith.maximumf %21, %cst_4 : f32
      %23 = arith.minimumf %22, %cst_5 : f32
      %24 = arith.fptosi %23 : f32 to i8
      linalg.yield %24 : i8
    } -> tensor<64x128xi8>

  return %13 : tensor<64x128xi8>
}
