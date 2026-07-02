// RUN: xdsl-opt %s -p gemmini-verify 2>&1 | FileCheck %s
//
// FUTURE: Verifies that when a tensor is not reused across iterations,
// the dialect does NOT mark it as resident (which would waste scratchpad space).
//
// This is a design correctness test, not a type-system rejection.
// The test checks that the analysis correctly identifies no-reuse patterns.
//
// CHECK-NOT: gemmini.resident_tensor{{.*}}no_reuse

func.func @no_reuse_should_not_pack(
    %A: memref<4x16x16xi8>,
    %B: memref<4x16x16xi8>,
    %C: memref<4x16x16xi32>
) {
  // TODO: Both A and B change each iteration; neither should be marked resident.
  // The dialect should emit gemmini.pack (non-resident) for both, not gemmini.pack_resident.
  return
}
