// RUN: xdsl-opt %s -p gemmini-verify | FileCheck %s
//
// FUTURE: Verifies that gemmini.commit accepts bias, scale, and relu flags
// matching the mvout requantisation pipeline in real Gemmini hardware.
//
// CHECK: gemmini.commit {{.*}}scale{{.*}}bias{{.*}}relu

func.func @matmul_bias_requant_relu(
    %A: memref<16x16xi8>,
    %B: memref<16x16xi8>,
    %bias: memref<16xi32>,
    %scale: f32,
    %C: memref<16x16xi8>
) {
  // TODO: replace with actual gemmini dialect ops.
  // Expected sequence:
  //   %a_tile = gemmini.pack %A {...} -> !gemmini.resident_tensor<16x16xi8>
  //   %b_tile = gemmini.pack %B {...} -> !gemmini.resident_tensor<16x16xi8>
  //   %acc    = gemmini.matmul %a_tile, %b_tile -> !gemmini.accumulator<16x16xi32>
  //   gemmini.commit %acc, %C {bias=%bias, scale=%scale, relu=true}
  //     : !gemmini.accumulator<16x16xi32>, memref<16xi32>, f32 -> memref<16x16xi8>
  return
}
