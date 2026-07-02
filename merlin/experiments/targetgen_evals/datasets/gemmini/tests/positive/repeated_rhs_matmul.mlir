// RUN: xdsl-opt %s -p gemmini-verify | FileCheck %s
//
// FUTURE: Verifies the resident_rhs optimisation:
//   - B is packed once before the loop
//   - The same resident_tensor is reused across all batch iterations
//   - The verifier must NOT reject this as a use-after-pack
//
// CHECK: gemmini.pack
// CHECK-COUNT-1: gemmini.pack {{.*}}B{{.*}}  -- B packed exactly once
// CHECK-COUNT-8: gemmini.matmul             -- 8 matmuls, one per batch

func.func @repeated_rhs_matmul(
    %A: memref<8x16x16xi8>,
    %B: memref<16x16xi8>,
    %C: memref<8x16x16xi32>
) {
  // TODO: replace with actual gemmini dialect ops.
  // Expected sequence:
  //   %b_resident = gemmini.pack %B {...} -> !gemmini.resident_tensor<16x16xi8>
  //   affine.for %i = 0 to 8 {
  //     %a_slice = memref.subview %A[%i, 0, 0][1, 16, 16][1, 1, 1]
  //     %c_slice = memref.subview %C[%i, 0, 0][1, 16, 16][1, 1, 1]
  //     %a_tile = gemmini.pack %a_slice {...} -> !gemmini.resident_tensor<16x16xi8>
  //     %acc    = gemmini.matmul %a_tile, %b_resident -> !gemmini.accumulator<16x16xi32>
  //     gemmini.commit %acc, %c_slice
  //   }
  return
}
