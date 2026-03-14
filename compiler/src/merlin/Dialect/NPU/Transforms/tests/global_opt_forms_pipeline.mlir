// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel)' | FileCheck %s --check-prefix=KERNEL
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,npu-verify-ukernel-symbols)' | FileCheck %s --check-prefix=VERIFY
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule)' | FileCheck %s --check-prefix=SCHEDULE
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,npu-verify-ukernel-symbols,convert-npu-kernel-to-schedule,npu-verify-ukernel-symbols,convert-npu-schedule-to-isa)' | FileCheck %s --check-prefix=ISA

#attn_q_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#attn_k_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#attn_v_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#attn_scale_map = affine_map<(d0, d1, d2, d3, d4) -> ()>
#attn_mask_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#attn_out_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>

module {
  func.func @batch_matmul_form(%lhs: tensor<2x4x16xbf16>, %rhs: tensor<2x16x8xbf16>) -> tensor<2x4x8xbf16> {
    %zero = arith.constant 0.0 : bf16
    %empty = tensor.empty() : tensor<2x4x8xbf16>
    %acc = linalg.fill ins(%zero : bf16) outs(%empty : tensor<2x4x8xbf16>) -> tensor<2x4x8xbf16>
    %0 = linalg.batch_matmul
      ins(%lhs, %rhs : tensor<2x4x16xbf16>, tensor<2x16x8xbf16>)
      outs(%acc : tensor<2x4x8xbf16>) -> tensor<2x4x8xbf16>
    return %0 : tensor<2x4x8xbf16>
  }

  func.func @softmax_form(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %empty = tensor.empty() : tensor<2x4xf32>
    %0 = linalg.softmax dimension(1) ins(%input : tensor<2x4xf32>) outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @attention_form(%q: tensor<1x2x4xbf16>, %k: tensor<1x2x4xbf16>, %v: tensor<1x2x4xbf16>, %mask: tensor<1x2x2xi1>) -> tensor<1x2x4xbf16> {
    %scale = arith.constant 0.5 : bf16
    %init = tensor.empty() : tensor<1x2x4xbf16>
    %0 = iree_linalg_ext.attention {
      indexing_maps = [#attn_q_map, #attn_k_map, #attn_v_map, #attn_scale_map, #attn_mask_map, #attn_out_map]
    } ins(%q, %k, %v, %scale, %mask : tensor<1x2x4xbf16>, tensor<1x2x4xbf16>, tensor<1x2x4xbf16>, bf16, tensor<1x2x2xi1>) outs(%init : tensor<1x2x4xbf16>) {
    ^bb0(%arg0: bf16):
      iree_linalg_ext.yield %arg0 : bf16
    } -> tensor<1x2x4xbf16>
    return %0 : tensor<1x2x4xbf16>
  }
}

// KERNEL-LABEL: func.func @batch_matmul_form
// KERNEL: npu_kernel.matmul

// KERNEL-LABEL: func.func @softmax_form
// KERNEL: npu_schedule.softmax_fragment

// KERNEL-LABEL: func.func @attention_form
// KERNEL: npu_kernel.ukernel_generic "npu_uk_gemma_attention_

// VERIFY: npu_kernel.ukernel_generic "npu_uk_gemma_attention_

// SCHEDULE-LABEL: func.func @batch_matmul_form
// SCHEDULE: npu_schedule.matmul_tile

// SCHEDULE-LABEL: func.func @softmax_form
// SCHEDULE: npu_schedule.softmax_fragment

// SCHEDULE-LABEL: func.func @attention_form
// SCHEDULE: npu_schedule.ukernel_launch "npu_uk_gemma_attention_

// ISA: npu_isa.matmul_mxu0
// ISA: npu_isa.vexp
