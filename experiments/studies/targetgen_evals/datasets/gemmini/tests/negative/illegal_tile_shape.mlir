// RUN: xdsl-opt %s -p gemmini-verify 2>&1 | FileCheck %s
//
// FUTURE: Verifies that a tile shape that is not a multiple of DIM=16 is rejected.
//
// CHECK: error: gemmini.matmul: tile_m=20 is not a multiple of DIM=16

func.func @illegal_tile_shape(
    %A: memref<20x16xi8>,
    %B: memref<16x16xi8>,
    %C: memref<20x16xi32>
) {
  // TODO: replace with actual dialect op that should be rejected.
  // gemmini.matmul {tile_m=20, tile_n=16, tile_k=16}  <- must fail: 20 % 16 != 0
  return
}
