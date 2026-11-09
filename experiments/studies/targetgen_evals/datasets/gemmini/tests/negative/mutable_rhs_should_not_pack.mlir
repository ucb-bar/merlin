// RUN: xdsl-opt %s -p gemmini-verify 2>&1 | FileCheck %s
//
// FUTURE: Verifies that a tensor with writes inside the loop body
// cannot be treated as a resident tensor (immutability invariant).
//
// CHECK: error: resident_tensor value is written inside loop body

func.func @mutable_rhs_should_not_pack(
    %A: memref<8x16x16xi8>,
    %B: memref<16x16xi8>,
    %C: memref<8x16x16xi32>
) {
  // TODO: If B is written inside the loop, pack_resident must be rejected.
  // The analysis or verifier should catch: resident_tensor must not alias a store.
  return
}
