// RUN: iree-opt --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=%t.mlir tile-m=128 tile-n=128 tile-k=64})" %s
// RUN: FileCheck --input-file=%t.mlir %s

// Test with custom tile sizes: tile-m=128, tile-n=128, tile-k=64.
// Matmul: 256x128 * 128x256.

func.func @matmul_f32(%A: tensor<256x128xf32>,
                       %B: tensor<128x256xf32>,
                       %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<256x128xf32>, tensor<128x256xf32>)
    outs(%C : tensor<256x256xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
  } -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// Verify custom tile sizes are reflected in the output.
// CHECK: TILE_M=128, TILE_N=128, TILE_K=64
// CHECK: cuda_tile.module @matmul_kernel
// CHECK: entry @matmul_f32

// With tile-k=64 and K=128: 128/64 = 2 K-tiles.
// CHECK: constant <i32: 2>

// Verify custom tile sizes in types.
// CHECK-DAG: tile<128x128xf32>
// CHECK-DAG: tile<128x64xf32>
// CHECK-DAG: tile<64x128xf32>

// Verify partition_view uses custom tile dims.
// CHECK-DAG: tile=(128x64)
// CHECK-DAG: tile=(64x128)
// CHECK-DAG: tile=(128x128)

// CHECK: mmaf
