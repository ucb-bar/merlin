// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-preprocess))' | FileCheck %s

// QDQ-cancelling max-pool fold: dequant + linalg.fill -inf + max-pool generic
// + quant, where dequant and quant scales bit-match, rewrites to a pure i8
// max-pool. Bit-exact for in-range i8 (verified end-to-end on FireSim Shuttle
// where dronet hash 0xd4d44793e1099c94 matches scalar with the fold enabled).

// CHECK-LABEL: func.func @pool_chain_no_transpose
func.func @pool_chain_no_transpose(%arg0: tensor<56x56x32xi8>) -> tensor<32x27x27xi8> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.0250296649 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant -1.280000e+02 : f32
  %cst_3 = arith.constant 1.270000e+02 : f32
  %1 = tensor.empty() : tensor<32x27x27xi8>
  %2 = tensor.empty() : tensor<32x27x27xf32>
  %3 = tensor.empty() : tensor<3x3xf32>
  %4 = tensor.empty() : tensor<56x56x32xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<56x56x32xi8>) outs(%4 : tensor<56x56x32xf32>) {
  ^bb0(%in: i8, %out: f32):
    %9 = arith.sitofp %in : i8 to f32
    %10 = arith.mulf %9, %cst_0 : f32
    linalg.yield %10 : f32
  } -> tensor<56x56x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x27x27xf32>) -> tensor<32x27x27xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d3, d2 * 2 + d4, d0)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %3 : tensor<56x56x32xf32>, tensor<3x3xf32>) outs(%6 : tensor<32x27x27xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.maximumf %out, %in : f32
    linalg.yield %9 : f32
  } -> tensor<32x27x27xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<32x27x27xf32>) outs(%1 : tensor<32x27x27xi8>) {
  ^bb0(%in: f32, %out: i8):
    %9 = arith.divf %in, %cst_0 : f32
    %10 = math.roundeven %9 : f32
    %11 = arith.addf %10, %cst_1 : f32
    %12 = arith.maximumf %11, %cst_2 : f32
    %13 = arith.minimumf %12, %cst_3 : f32
    %14 = arith.fptosi %13 : f32 to i8
    linalg.yield %14 : i8
  } -> tensor<32x27x27xi8>
  return %8 : tensor<32x27x27xi8>
}

// CHECK: %[[NEG128:.+]] = arith.constant -128 : i8
// CHECK: linalg.fill ins(%[[NEG128]] : i8)
// CHECK: linalg.generic
// CHECK:   arith.maxsi
// CHECK-NOT: arith.maximumf
// CHECK-NOT: arith.sitofp
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.divf

// -----

// Same chain but with a permutation-only transpose between dequant and pool
// (the pre-bufferization form IREE emits for dronet's dispatch_3). The
// matcher must walk through the transpose and compose its indexing map with
// the pool's window map so the new i8 pool reads from the original HWC
// tensor directly.

// CHECK-LABEL: func.func @pool_chain_with_transpose
func.func @pool_chain_with_transpose(%arg0: tensor<56x56x32xi8>) -> tensor<32x27x27xi8> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.0250296649 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant -1.280000e+02 : f32
  %cst_3 = arith.constant 1.270000e+02 : f32
  %1 = tensor.empty() : tensor<32x27x27xi8>
  %2 = tensor.empty() : tensor<32x27x27xf32>
  %3 = tensor.empty() : tensor<3x3xf32>
  %4 = tensor.empty() : tensor<56x56x32xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<56x56x32xi8>) outs(%4 : tensor<56x56x32xf32>) {
  ^bb0(%in: i8, %out: f32):
    %9 = arith.sitofp %in : i8 to f32
    %10 = arith.mulf %9, %cst_0 : f32
    linalg.yield %10 : f32
  } -> tensor<56x56x32xf32>
  %chw_init = tensor.empty() : tensor<32x56x56xf32>
  // HWC -> CHW transpose: out[d0,d1,d2] = in[d1,d2,d0]
  %chw_val = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<56x56x32xf32>) outs(%chw_init : tensor<32x56x56xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<32x56x56xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x27x27xf32>) -> tensor<32x27x27xf32>
  // Pool reads CHW input with window stride 2.
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2 + d3, d2 * 2 + d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%chw_val, %3 : tensor<32x56x56xf32>, tensor<3x3xf32>) outs(%6 : tensor<32x27x27xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.maximumf %out, %in : f32
    linalg.yield %9 : f32
  } -> tensor<32x27x27xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<32x27x27xf32>) outs(%1 : tensor<32x27x27xi8>) {
  ^bb0(%in: f32, %out: i8):
    %9 = arith.divf %in, %cst_0 : f32
    %10 = math.roundeven %9 : f32
    %11 = arith.addf %10, %cst_1 : f32
    %12 = arith.maximumf %11, %cst_2 : f32
    %13 = arith.minimumf %12, %cst_3 : f32
    %14 = arith.fptosi %13 : f32 to i8
    linalg.yield %14 : i8
  } -> tensor<32x27x27xi8>
  return %8 : tensor<32x27x27xi8>
}

// The composed window map for the i8 pool reads the original HWC tensor
// (transpose folded into pool indexing). Check that the rewrite consumed
// `%arg0` (the i8 HWC tensor) directly — proving the transpose was walked
// through — and that the body is the i8 maxsi form.
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0,
// CHECK:   arith.maxsi
// CHECK-NOT: arith.maximumf

// -----

// Negative case: dequant and quant scales differ. The fold MUST be declined
// (otherwise we'd silently break numerics).

// CHECK-LABEL: func.func @pool_chain_scale_mismatch
func.func @pool_chain_scale_mismatch(%arg0: tensor<56x56x32xi8>) -> tensor<32x27x27xi8> {
  %cst = arith.constant 0xFF800000 : f32
  %cst_in = arith.constant 0.0250296649 : f32
  %cst_out = arith.constant 0.0500593298 : f32  // different scale
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant -1.280000e+02 : f32
  %cst_3 = arith.constant 1.270000e+02 : f32
  %1 = tensor.empty() : tensor<32x27x27xi8>
  %2 = tensor.empty() : tensor<32x27x27xf32>
  %3 = tensor.empty() : tensor<3x3xf32>
  %4 = tensor.empty() : tensor<56x56x32xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<56x56x32xi8>) outs(%4 : tensor<56x56x32xf32>) {
  ^bb0(%in: i8, %out: f32):
    %9 = arith.sitofp %in : i8 to f32
    %10 = arith.mulf %9, %cst_in : f32
    linalg.yield %10 : f32
  } -> tensor<56x56x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x27x27xf32>) -> tensor<32x27x27xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d3, d2 * 2 + d4, d0)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %3 : tensor<56x56x32xf32>, tensor<3x3xf32>) outs(%6 : tensor<32x27x27xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %9 = arith.maximumf %out, %in : f32
    linalg.yield %9 : f32
  } -> tensor<32x27x27xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<32x27x27xf32>) outs(%1 : tensor<32x27x27xi8>) {
  ^bb0(%in: f32, %out: i8):
    %9 = arith.divf %in, %cst_out : f32
    %10 = math.roundeven %9 : f32
    %11 = arith.addf %10, %cst_1 : f32
    %12 = arith.maximumf %11, %cst_2 : f32
    %13 = arith.minimumf %12, %cst_3 : f32
    %14 = arith.fptosi %13 : f32 to i8
    linalg.yield %14 : i8
  } -> tensor<32x27x27xi8>
  return %8 : tensor<32x27x27xi8>
}

// CHECK: arith.maximumf
// CHECK: arith.divf
// CHECK-NOT: arith.maxsi
