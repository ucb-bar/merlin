// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | FileCheck %s

module {
  func.func @matmul(%lhs: tensor<64x32xbf16>, %rhs: tensor<32x16xbf16>) -> tensor<64x16xbf16> {
    %init = tensor.empty() : tensor<64x16xbf16>
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<64x32xbf16>, tensor<32x16xbf16>)
        outs(%init : tensor<64x16xbf16>)
      -> tensor<64x16xbf16>
    return %0 : tensor<64x16xbf16>
  }
}

// CHECK: npu_isa.dma_load
// CHECK: npu_isa.dma_wait
// CHECK: npu_isa.dma_load_mxu0
// CHECK: npu_isa.matmul_mxu0
