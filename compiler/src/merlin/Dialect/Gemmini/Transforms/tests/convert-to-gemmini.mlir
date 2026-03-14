// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-convert-to-gemmini))' | FileCheck %s

#lhs_map = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs_map = affine_map<(d0, d1, d2) -> (d1, d2)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_i8(%lhs: tensor<16x1024xi8>, %rhs: tensor<128x1024xi8>) -> tensor<16x128xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x128xi32>
  %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<16x128xi32>) -> tensor<16x128xi32>
  %0 = linalg.generic
      {indexing_maps = [
         affine_map<(d0, d1, d2) -> (d0, d2)>,
         affine_map<(d0, d1, d2) -> (d1, d2)>,
         affine_map<(d0, d1, d2) -> (d0, d1)>
       ],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<16x1024xi8>, tensor<128x1024xi8>)
      outs(%init : tensor<16x128xi32>) {
    ^bb0(%in0: i8, %in1: i8, %acc: i32):
      %0 = arith.extsi %in0 : i8 to i32
      %1 = arith.extsi %in1 : i8 to i32
      %2 = arith.muli %0, %1 : i32
      %3 = arith.addi %acc, %2 : i32
      linalg.yield %3 : i32
  } -> tensor<16x128xi32>
  return %0 : tensor<16x128xi32>
}

// CHECK-LABEL: func.func @matmul_i8
// CHECK: gemmini.matmul

func.func @matmul_named_i8(%lhs: tensor<16x32xi8>, %rhs: tensor<64x32xi8>) -> tensor<16x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<16x64xi32>
  %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<16x64xi32>) -> tensor<16x64xi32>
  %0 = linalg.matmul indexing_maps = [#lhs_map, #rhs_map, #out_map]
      ins(%lhs, %rhs : tensor<16x32xi8>, tensor<64x32xi8>)
      outs(%init : tensor<16x64xi32>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>
}

// CHECK-LABEL: func.func @matmul_named_i8
// CHECK: gemmini.matmul
