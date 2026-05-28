// 1x1x2048 int8 -> i32 matmul. Same shape as dronet's linear1/linear2 FC
// heads (steer / collision). M=1, J=1, K=2048. Highly anisotropic — padded
// to dim=16 means padI=15, padJ=15, padK=0. tileI=1, tileJ=1, tileK=128.
//
// Use this fixture to repro the 2026-05-17 Gemmini steer-vs-collision
// numerical bug in a STANDALONE environment without the full dronet
// pipeline. Both runs (with two different B tensors) should give an i32
// that matches `numpy.matmul(A_i32, B_i32)`. If Gemmini is wrong by 8x
// for one B tensor and correct for the other, we have the bug reproduced
// outside dronet — at which point bisecting B is fast.
//
// See: project_gemmini_steer_8x_bug.md, project_gemmini_numerical_bug_2026_05_17.md

func.func @matmul_1x1x2048(%A: memref<1x2048xi8>, %B: memref<2048x1xi8>, %C: memref<1x1xi32>)
    attributes {iree.preserve_func_visibility = true} {
  linalg.matmul ins(%A, %B : memref<1x2048xi8>, memref<2048x1xi8>)
                outs(%C : memref<1x1xi32>)
  return
}
