// RUN: xdsl-opt %s -p gemmini-verify | FileCheck %s
// RUN: xdsl-opt %s -p gemmini-to-runtime-commands | FileCheck %s --check-prefix=LOWER
//
// FUTURE: Once xdsl-opt is wired in, this test verifies that a minimal
// 16x16x16 i8 matmul using the gemmini dialect is accepted by the verifier
// and lowers to a valid runtime command sequence.
//
// CHECK: gemmini.matmul
// LOWER: runtime_cmd.dispatch

func.func @matmul_16x16x16(
    %A: memref<16x16xi8>,
    %B: memref<16x16xi8>,
    %C: memref<16x16xi32>
) {
  // TODO: replace with actual gemmini dialect ops once generated.
  // Expected sequence:
  //   %a_tile = gemmini.pack %A {tile_m=16, tile_k=16} : memref<16x16xi8> -> !gemmini.resident_tensor<16x16xi8>
  //   %b_tile = gemmini.pack %B {tile_k=16, tile_n=16} : memref<16x16xi8> -> !gemmini.resident_tensor<16x16xi8>
  //   %acc    = gemmini.matmul %a_tile, %b_tile : ... -> !gemmini.accumulator<16x16xi32>
  //   gemmini.commit %acc, %C : !gemmini.accumulator<16x16xi32>, memref<16x16xi32>
  return
}
