// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(tile-npu-kernel-to-schedule)' | FileCheck %s

// A 32x64 @ 64x32 matmul has K = 64 = 2 K-tiles. The pass should split it
// into one M/N iteration containing two unrolled npu_schedule.ukernel_launch
// invocations: matmul_acc_first (k=0) followed by matmul_acc_last (k=1).

module {
  func.func @matmul_two_k_tiles(
      %lhs: tensor<32x64xf8E4M3FN>,
      %rhs: tensor<64x32xf8E4M3FN>) -> tensor<32x32xbf16> {
    %0 = npu_kernel.matmul %lhs, %rhs
        : tensor<32x64xf8E4M3FN>, tensor<64x32xf8E4M3FN> -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

// Tile-loop scaffolding (M and N collapse to 1 iteration each, but the loop
// is still emitted; canonicalization can fold them later if needed).
// CHECK: scf.for
// CHECK: scf.for

// First K iteration writes the accumulator with no preexisting state.
// CHECK-DAG: npu_schedule.ukernel_launch "npu_uk_matmul_acc_first"

// Last K iteration drains the accumulator and produces the result tile.
// CHECK-DAG: npu_schedule.ukernel_launch "npu_uk_matmul_acc_last"

// No "_mid" variant for K = 2 tiles.
// CHECK-NOT: "npu_uk_matmul_acc_mid"

// -----

// 32x96 @ 96x32 → K = 3 tiles → first / mid / last triple.
module {
  func.func @matmul_three_k_tiles(
      %lhs: tensor<32x96xf8E4M3FN>,
      %rhs: tensor<96x32xf8E4M3FN>) -> tensor<32x32xbf16> {
    %0 = npu_kernel.matmul %lhs, %rhs
        : tensor<32x96xf8E4M3FN>, tensor<96x32xf8E4M3FN> -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

// CHECK-LABEL: @matmul_three_k_tiles
// CHECK-DAG: "npu_uk_matmul_acc_first"
// CHECK-DAG: "npu_uk_matmul_acc_mid"
// CHECK-DAG: "npu_uk_matmul_acc_last"
