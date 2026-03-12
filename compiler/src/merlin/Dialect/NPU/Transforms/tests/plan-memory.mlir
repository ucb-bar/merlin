// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(npu-plan-memory)' | FileCheck %s

module {
  func.func @plan(%arg0: tensor<64x32xf8E4M3FN>, %arg1: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    npu_isa.dma_load rd = 2, base = 111, size = 2048, flag = 9
    npu_isa.dma_wait flag = 99
    npu_isa.dma_load_mxu0 rd = 1, base = 222, size = 512, flag = 9
    npu_isa.dma_wait flag = 98
    %0 = npu_isa.matmul_mxu0 %arg0, %arg1 regs = (0, 2, 1)
      : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    npu_isa.dma_store %0 rs1 = 0, base = 333, size = 1024, flag = 9 : tensor<64x16xf32>
    npu_isa.dma_wait flag = 97
    return %0 : tensor<64x16xf32>
  }
}

// CHECK-LABEL: func.func @plan
// CHECK: npu_isa.dma_load rd = 2, base = 0, size = 2048, flag = 0
// CHECK: npu_isa.dma_wait flag = 0
// CHECK: npu_isa.dma_load_mxu0 rd = 1, base = 8192, size = 512, flag = 1
// CHECK: npu_isa.dma_wait flag = 1
// CHECK: npu_isa.dma_store {{.*}} base = 20480, size = 1024, flag = 2
// CHECK: npu_isa.dma_wait flag = 2
