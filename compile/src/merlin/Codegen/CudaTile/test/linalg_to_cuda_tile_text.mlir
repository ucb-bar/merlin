// RUN: iree-opt --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=%t.mlir})" %s 2>&1 | FileCheck %s --check-prefix=REMARK
// RUN: FileCheck --input-file=%t.mlir %s

// Verify that the LinalgToCudaTileText pass emits correct cuda_tile-dialect
// textual IR for a simple matmul linalg.generic with default tile sizes
// (tile-m=64, tile-n=64, tile-k=32).

func.func @matmul_f32(%A: tensor<128x64xf32>,
                       %B: tensor<64x256xf32>,
                       %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<128x64xf32>, tensor<64x256xf32>)
    outs(%C : tensor<128x256xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %mul, %c : f32
    linalg.yield %add : f32
  } -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// --- Check the diagnostic remark ---
// REMARK: remark: lowering matmul C[128x256] = A[128x64] * B[64x256] with tiles (64,64,32)

// --- Checks on the emitted cuda_tile text file ---

// Verify top-level cuda_tile.module (no builtin module wrapper).
// CHECK: cuda_tile.module @matmul_kernel
// CHECK: entry @matmul_f32

// Verify scalar pointer entry args.
// CHECK: tile<ptr<f32>>

// Verify tensor_view and partition_view setup.
// CHECK: make_tensor_view
// CHECK: make_partition_view

// Verify tile block ID.
// CHECK: get_tile_block_id

// Verify load/mmaf/store ops.
// CHECK: load_view_tko weak
// CHECK: mmaf
// CHECK: store_view_tko weak

// Verify the for loop with continue terminator.
// CHECK: for %k in
// CHECK: continue

// Verify tile sizes appear in the generated types.
// CHECK-DAG: tile<64x64xf32>
// CHECK-DAG: tile<64x32xf32>
// CHECK-DAG: tile<32x64xf32>

// Verify the dimension comment.
// CHECK: C[128x256] = A[128x64] * B[64x256]

// Verify tile sizes in the comment.
// CHECK: TILE_M=64, TILE_N=64, TILE_K=32
