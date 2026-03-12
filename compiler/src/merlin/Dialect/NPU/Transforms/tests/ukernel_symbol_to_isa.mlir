// RUN: iree-opt --split-input-file %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-npu-kernel-to-schedule)' | FileCheck %s --check-prefix=SCHEDULE
// RUN: iree-opt --split-input-file %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | FileCheck %s --check-prefix=ISA
// RUN: iree-opt --split-input-file %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | npu-translate --mlir-to-npu-text-isa | FileCheck %s --check-prefix=TEXT
// RUN: iree-opt --split-input-file %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-npu-schedule-to-isa)' | FileCheck %s --check-prefix=SOFTMAX

module {
  func.func @uk_matmul_sym(%lhs: tensor<64x32xf8E4M3FN>, %rhs: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %0 = npu_kernel.ukernel_generic "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"(%lhs, %rhs) : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

// SCHEDULE-LABEL: func.func @uk_matmul_sym
// SCHEDULE: npu_schedule.ukernel_launch "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"
// ISA-LABEL: func.func @uk_matmul_sym
// ISA: npu_isa.dma_load
// ISA: npu_isa.dma_wait
// ISA: npu_isa.dma_load_mxu0
// ISA: npu_isa.matmul_mxu0
// TEXT: dma.load.mxu0 rd=1, base=2048, size=512, flag=0
// TEXT: matmul.mxu0 rd=0, rs1=2, rs2=1

// -----

module {
  func.func @uk_gemma_mlp_sym(%lhs: tensor<64x32xf8E4M3FN>, %rhs: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %0 = npu_kernel.ukernel_generic "npu_uk_gemma_mlp_f8E4M3FN_f8E4M3FN_f32"(%lhs, %rhs) : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

// SCHEDULE-LABEL: func.func @uk_gemma_mlp_sym
// SCHEDULE: npu_schedule.ukernel_launch "npu_uk_gemma_mlp_f8E4M3FN_f8E4M3FN_f32"
// ISA-LABEL: func.func @uk_gemma_mlp_sym
// ISA: npu_isa.dma_load_mxu1
// ISA: npu_isa.matmul_mxu0
// ISA: npu_isa.vmul
// ISA: npu_isa.dma_store
// TEXT: dma.load.mxu1 rd=0, base=0, size=512, flag=0
// TEXT: dma.load.mxu1 rd=1, base=512, size=512, flag=1
// TEXT: vmul vrd=6, vs1=1, vs2=2
// TEXT: dma.store rs1=6, base=12288, size=2048, flag=0

// -----

module {
  func.func @uk_gemma_attention_sym(%q: tensor<64x16xf8E4M3FN>, %kv: tensor<16x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %0 = npu_kernel.ukernel_generic "npu_uk_gemma_attention_f8E4M3FN_f8E4M3FN_f32"(%q, %kv) : tensor<64x16xf8E4M3FN>, tensor<16x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

// SCHEDULE-LABEL: func.func @uk_gemma_attention_sym
// SCHEDULE: npu_schedule.ukernel_launch "npu_uk_gemma_attention_f8E4M3FN_f8E4M3FN_f32"
// ISA-LABEL: func.func @uk_gemma_attention_sym
// ISA: npu_isa.dma_load_mxu1
// ISA: npu_isa.matmul_mxu0
// ISA: npu_isa.vexp
// ISA: npu_isa.vreduce_sum
// ISA: npu_isa.vrcp
// ISA: npu_isa.dma_store
// TEXT: vexp vrd=5, vs1=4
// TEXT: vreduce.sum vrd=6, vs1=5
// TEXT: vrcp vrd=7, vs1=6
// TEXT: dma.store rs1=9, base=20480, size=2048, flag=1

// -----

module {
  func.func @schedule_softmax_direct(%arg0: tensor<64x16xf32>) -> tensor<64x16xf32> {
    %0 = npu_schedule.softmax_fragment %arg0 : tensor<64x16xf32> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

// SOFTMAX-LABEL: func.func @schedule_softmax_direct
// SOFTMAX: npu_isa.vmul
// SOFTMAX: npu_isa.vexp
// SOFTMAX: npu_isa.vreduce_sum
// SOFTMAX: npu_isa.vrcp
