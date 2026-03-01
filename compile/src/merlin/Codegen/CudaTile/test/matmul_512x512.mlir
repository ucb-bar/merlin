// RUN: iree-opt --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=%t.mlir})" %s
// RUN: FileCheck --input-file=%t.mlir %s

// Large matmul: 512x128 * 128x512, default tiles (64/64/32).

func.func @matmul_f32(%A: tensor<512x128xf32>,
                       %B: tensor<128x512xf32>,
                       %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<512x128xf32>, tensor<128x512xf32>)
    outs(%C : tensor<512x512xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
  } -> tensor<512x512xf32>
  return %result : tensor<512x512xf32>
}

// CHECK: C[512x512] = A[512x128] * B[128x512]
// CHECK: cuda_tile.module @matmul_kernel
// CHECK: entry @matmul_f32
// CHECK: make_tensor_view %A_base, shape = [512, 128]
// CHECK: make_tensor_view %B_base, shape = [128, 512]
// K=128 / TILE_K=32 = 4 tiles
// CHECK: constant <i32: 4>
// CHECK-DAG: tile<64x64xf32>
// CHECK-DAG: tile<64x32xf32>
// CHECK-DAG: tile<32x64xf32>
// CHECK: mmaf
