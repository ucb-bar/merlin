// RUN: iree-opt --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=%t.mlir})" %s
// RUN: FileCheck --input-file=%t.mlir %s

// Minimal matmul: 64x32 * 32x64 — exactly 1 tile per dim with default tiles.

func.func @matmul_f32(%A: tensor<64x32xf32>,
                       %B: tensor<32x64xf32>,
                       %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
    outs(%C : tensor<64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
  } -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// CHECK: C[64x64] = A[64x32] * B[32x64]
// CHECK: cuda_tile.module @matmul_kernel
// CHECK: entry @matmul_f32
// K=32 / TILE_K=32 = 1 tile
// CHECK: constant <i32: 1>
// CHECK-DAG: tile<64x64xf32>
// CHECK-DAG: tile<64x32xf32>
// CHECK-DAG: tile<32x64xf32>
// CHECK: mmaf
