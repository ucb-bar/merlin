// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-lower-gemmini-to-iree))' | FileCheck %s

func.func @lower_tile_to_linalg(%lhs: tensor<16x32xi8>, %rhs: tensor<64x32xi8>) -> tensor<16x64xi32> {
  %0 = gemmini.matmul_tile %lhs, %rhs {dataflow = #gemmini.dataflow<os>, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64, tileM = 16 : i64, tileN = 16 : i64, tileK = 16 : i64} : tensor<16x32xi8>, tensor<64x32xi8> -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>
}

func.func @lower_tile_fp8_to_linalg(%lhs: tensor<16x32xf8E4M3FN>, %rhs: tensor<64x32xf8E4M3FN>) -> tensor<16x64xbf16> {
  %0 = gemmini.matmul_tile %lhs, %rhs {dataflow = #gemmini.dataflow<ws>, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64, tileM = 16 : i64, tileN = 16 : i64, tileK = 16 : i64} : tensor<16x32xf8E4M3FN>, tensor<64x32xf8E4M3FN> -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// CHECK-LABEL: func.func @lower_tile_to_linalg
// CHECK: linalg.generic
// CHECK-NOT: gemmini.
// CHECK-LABEL: func.func @lower_tile_fp8_to_linalg
// CHECK: linalg.generic
// CHECK: arith.extf
// CHECK-NOT: gemmini.
