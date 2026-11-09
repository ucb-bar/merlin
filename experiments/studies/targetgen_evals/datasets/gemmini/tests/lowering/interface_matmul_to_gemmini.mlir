// RUN: xdsl-opt %s -p interface-matmul-to-gemmini | FileCheck %s
//
// FUTURE: Verifies that a generic linalg.matmul on a Gemmini-targeted tensor
// is lowered to the gemmini dialect by the interface lowering pass.
//
// This tests the Merlin interface dialect → gemmini dialect bridge.
// It does NOT test gemmini → runtime_command (that is tested in positive/).
//
// CHECK: gemmini.pack
// CHECK: gemmini.matmul
// CHECK: gemmini.commit

func.func @interface_matmul_to_gemmini(
    %A: memref<16x16xi8>,
    %B: memref<16x16xi8>,
    %C: memref<16x16xi32>
) {
  // TODO: Replace with actual interface.matmul op targeting the gemmini backend.
  // interface.matmul %A, %B, %C {target="gemmini", tile_m=16, tile_n=16, tile_k=16}
  //   : memref<16x16xi8>, memref<16x16xi8>, memref<16x16xi32>
  return
}
