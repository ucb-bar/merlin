// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(tile-npu-kernel-to-schedule)' | FileCheck %s

// A 2D attention with (seq_q = 32, head = 32) Q and (seq_kv = 64, head = 32)
// K/V → 2 K/V tiles → first/last variant pair.
module {
  func.func @attention_two_kv_tiles(
      %q: tensor<32x32xbf16>,
      %k: tensor<64x32xbf16>,
      %v: tensor<64x32xbf16>) -> tensor<32x32xbf16> {
    %0 = npu_kernel.ukernel_generic "npu_uk_attention"(%q, %k, %v)
        : tensor<32x32xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>
        -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}

// CHECK-LABEL: @attention_two_kv_tiles
// CHECK: scf.for
// CHECK-DAG: "npu_uk_attention_acc_first"
// CHECK-DAG: "npu_uk_attention_acc_last"
// CHECK-NOT: "npu_uk_attention_acc_mid"

// -----

// A 3D batched attention with (batch=2, seq_q=32, head=32) and seq_kv=96 →
// 3 K/V tiles → first / mid / last triple, and a batch loop wrapping it.
module {
  func.func @attention_batched_three_kv(
      %q: tensor<2x32x32xbf16>,
      %k: tensor<2x96x32xbf16>,
      %v: tensor<2x96x32xbf16>) -> tensor<2x32x32xbf16> {
    %0 = npu_kernel.ukernel_generic "npu_uk_attention"(%q, %k, %v)
        : tensor<2x32x32xbf16>, tensor<2x96x32xbf16>, tensor<2x96x32xbf16>
        -> tensor<2x32x32xbf16>
    return %0 : tensor<2x32x32xbf16>
  }
}

// CHECK-LABEL: @attention_batched_three_kv
// CHECK: scf.for
// CHECK: scf.for
// CHECK-DAG: "npu_uk_attention_acc_first"
// CHECK-DAG: "npu_uk_attention_acc_mid"
// CHECK-DAG: "npu_uk_attention_acc_last"
