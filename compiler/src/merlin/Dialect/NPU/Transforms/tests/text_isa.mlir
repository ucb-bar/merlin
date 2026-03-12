// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | npu-translate --mlir-to-npu-text-isa | FileCheck %s

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

// CHECK: dma.load rd=2, base=0, size=2048, flag=0
// CHECK: dma.wait flag=0
// CHECK: dma.load.mxu0 rd=1, base=2048, size=512, flag=0
// CHECK: dma.wait flag=0
// CHECK: matmul.mxu0 rd=0, rs1=2, rs2=1
