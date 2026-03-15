// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-lower-to-isa))' | FileCheck %s

func.func @matmul_to_tile(%lhs: tensor<16x32xi8>, %rhs: tensor<64x32xi8>) -> tensor<16x64xi32> {
  %0 = gemmini.matmul %lhs, %rhs {dataflow = #gemmini.dataflow<ws>, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64} : tensor<16x32xi8>, tensor<64x32xi8> -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>
}

func.func @matmul_fp8_to_tile(%lhs: tensor<16x32xf8E4M3FN>, %rhs: tensor<64x32xf8E4M3FN>) -> tensor<16x64xbf16> {
  %0 = gemmini.matmul %lhs, %rhs {dataflow = #gemmini.dataflow<ws>, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64} : tensor<16x32xf8E4M3FN>, tensor<64x32xf8E4M3FN> -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// CHECK-LABEL: func.func @matmul_to_tile
// CHECK: gemmini.matmul_tile
// CHECK-LABEL: func.func @matmul_fp8_to_tile
// CHECK: gemmini.matmul_tile
