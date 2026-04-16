// Phase D-gap-7: representative chunk extracted from vit_small d9
// (matmul + residual + bias + multi-round saturation requant).
// Lifted from
//   build/compiled_models/.../saturn_opu_OPU_opu_bench_vit_small.q.int8/
//     sources/module_main_graph$async_dispatch_9.mlir:32-81
//
// Structure (mirrors d9 exactly):
//   i8×i8 matmul as linalg.generic (d0,d2)×(d1,d2)→(d0,d1)
//   residual i8 → f32 scaled dequant
//   4-in 1-out fused generic: residual + matmul*scale + bias → i8
//     with triple saturation-clamp requant (the bit that has me worried).
func.func @main(%x: tensor<64x128xf32>) -> tensor<64x128xi8> {
  %cst    = arith.constant 0.015625 : f32   // matmul out scale (d9 %cst)
  %cst_0  = arith.constant 0.0625 : f32     // quant step (d9 %cst_0)
  %cst_1  = arith.constant 0.0 : f32        // zero point (d9 %cst_1)
  %cst_2  = arith.constant -128.0 : f32     // i8 min (d9 %cst_2)
  %cst_3  = arith.constant 127.0 : f32      // i8 max (d9 %cst_3)
  %cst_4  = arith.constant 0.03125 : f32    // requant step (d9 %cst_4)
  %cst_5  = arith.constant 0.0078125 : f32  // residual scale (d9 %cst_5)
  %c0_i32 = arith.constant 0 : i32

  // Synthesize i8 LHS from f32 harness input.
  %lhs_init = tensor.empty() : tensor<64x128xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<64x128xf32>) outs(%lhs_init : tensor<64x128xi8>) {
    ^bb0(%in: f32, %out: i8):
      %c = arith.fptosi %in : f32 to i8
      linalg.yield %c : i8
    } -> tensor<64x128xi8>

  // Constants for rhs, residual, bias.
  %6 = arith.constant dense<2>   : tensor<128x128xi8>
  %7 = arith.constant dense<3>   : tensor<64x128xi8>
  %8 = arith.constant dense<0.1> : tensor<128xf32>

  // Matmul i8×i8 → i32 (linalg.generic with d9's exact indexing maps).
  %9 = tensor.empty() : tensor<64x128xi32>
  %10 = linalg.fill ins(%c0_i32 : i32) outs(%9 : tensor<64x128xi32>) -> tensor<64x128xi32>
  %11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%5, %6 : tensor<64x128xi8>, tensor<128x128xi8>) outs(%10 : tensor<64x128xi32>) {
    ^bb0(%in: i8, %in_6: i8, %out: i32):
      %17 = arith.extsi %in : i8 to i32
      %18 = arith.extsi %in_6 : i8 to i32
      %19 = arith.muli %17, %18 : i32
      %20 = arith.addi %out, %19 : i32
      linalg.yield %20 : i32
    } -> tensor<64x128xi32>

  // Residual i8 → f32 (dequant).
  %12 = tensor.empty() : tensor<64x128xf32>
  %13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%7 : tensor<64x128xi8>) outs(%12 : tensor<64x128xf32>) {
    ^bb0(%in: i8, %out: f32):
      %17 = arith.sitofp %in : i8 to f32
      %18 = arith.mulf %17, %cst_5 : f32
      linalg.yield %18 : f32
    } -> tensor<64x128xf32>

  // Fused residual + matmul*scale + bias → i8 with d9's exact multi-round
  // saturation-clamp requant chain.
  %14 = tensor.empty() : tensor<64x128xi8>
  %15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%13, %11, %8 : tensor<64x128xf32>, tensor<64x128xi32>, tensor<128xf32>)
    outs(%14 : tensor<64x128xi8>) {
    ^bb0(%in: f32, %in_6: i32, %in_7: f32, %out: i8):
      %17 = arith.sitofp %in_6 : i32 to f32
      %18 = arith.mulf %17, %cst : f32
      %19 = arith.addf %18, %in_7 : f32
      %20 = arith.divf %19, %cst_0 : f32
      %21 = math.roundeven %20 : f32
      %22 = arith.addf %21, %cst_1 : f32
      %23 = arith.maximumf %22, %cst_2 : f32
      %24 = arith.minimumf %23, %cst_3 : f32
      %25 = arith.fptosi %24 : f32 to i8
      %26 = arith.sitofp %25 : i8 to f32
      %27 = math.roundeven %26 : f32
      %28 = arith.addf %27, %cst_1 : f32
      %29 = arith.maximumf %28, %cst_2 : f32
      %30 = arith.minimumf %29, %cst_3 : f32
      %31 = arith.fptosi %30 : f32 to i8
      %32 = arith.sitofp %31 : i8 to f32
      %33 = math.roundeven %32 : f32
      %34 = arith.addf %33, %cst_1 : f32
      %35 = arith.maximumf %34, %cst_2 : f32
      %36 = arith.minimumf %35, %cst_3 : f32
      %37 = arith.fptosi %36 : f32 to i8
      %38 = arith.sitofp %37 : i8 to f32
      %39 = arith.mulf %38, %cst_0 : f32
      %40 = arith.addf %in, %39 : f32
      %41 = arith.divf %40, %cst_4 : f32
      %42 = math.roundeven %41 : f32
      %43 = arith.addf %42, %cst_1 : f32
      %44 = arith.maximumf %43, %cst_2 : f32
      %45 = arith.minimumf %44, %cst_3 : f32
      %46 = arith.fptosi %45 : f32 to i8
      linalg.yield %46 : i8
    } -> tensor<64x128xi8>

  return %15 : tensor<64x128xi8>
}
