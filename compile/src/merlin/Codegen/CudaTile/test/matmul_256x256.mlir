// RUN: iree-opt --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=%t.mlir})" %s
// RUN: FileCheck --input-file=%t.mlir %s

// Square matmul: 256x256 * 256x256, default tiles (64/64/32).

func.func @matmul_f32(%A: tensor<256x256xf32>,
                       %B: tensor<256x256xf32>,
                       %C: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%C : tensor<256x256xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
  } -> tensor<256x256xf32>
  return %result : tensor<256x256xf32>
}

// CHECK: C[256x256] = A[256x256] * B[256x256]
// CHECK: TILE_M=64, TILE_N=64, TILE_K=32
// CHECK: cuda_tile.module @matmul_kernel
// CHECK: entry @matmul_f32
// CHECK: make_tensor_view %A_base, shape = [256, 256]
// CHECK: make_tensor_view %B_base, shape = [256, 256]
// CHECK: make_tensor_view %C_base, shape = [256, 256]
// K=256 / TILE_K=32 = 8 tiles
// CHECK: constant <i32: 8>
// CHECK-DAG: tile<64x64xf32>
// CHECK-DAG: tile<64x32xf32>
// CHECK-DAG: tile<32x64xf32>
// CHECK: mmaf
