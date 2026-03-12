// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel)' | FileCheck %s --check-prefix=KERNEL
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule)' | FileCheck %s --check-prefix=SCHEDULE
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | FileCheck %s --check-prefix=ISA
// RUN: iree-opt %s --iree-plugin=npu --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' | npu-translate --mlir-to-npu-text-isa | FileCheck %s --check-prefix=TEXT

#lhs_map = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs_map = affine_map<(d0, d1, d2) -> (d2, d1)>
#out_map = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_generic_f8f8f32(%lhs: tensor<64x32xf8E4M3FN>, %rhs: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x16xf32>
    %acc = linalg.fill ins(%zero : f32) outs(%init : tensor<64x16xf32>) -> tensor<64x16xf32>
    %result = linalg.generic {
      indexing_maps = [#lhs_map, #rhs_map, #out_map],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN>)
      outs(%acc : tensor<64x16xf32>) {
    ^bb0(%a: f8E4M3FN, %b: f8E4M3FN, %c: f32):
      %af = arith.extf %a : f8E4M3FN to f32
      %bf = arith.extf %b : f8E4M3FN to f32
      %mul = arith.mulf %af, %bf : f32
      %sum = arith.addf %c, %mul : f32
      linalg.yield %sum : f32
    } -> tensor<64x16xf32>
    return %result : tensor<64x16xf32>
  }
}

// KERNEL: npu_kernel.ukernel_generic
// KERNEL-SAME: "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"

// SCHEDULE: npu_schedule.ukernel_launch
// SCHEDULE-SAME: "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"

// ISA: npu_isa.dma_load
// ISA: npu_isa.dma_wait
// ISA: npu_isa.dma_load_mxu0
// ISA: npu_isa.matmul_mxu0

// TEXT: dma.load rd=2, base=0, size=2048, flag=0
// TEXT: dma.wait flag=0
// TEXT: dma.load.mxu0 rd=1, base=2048, size=512, flag=0
// TEXT: dma.wait flag=0
// TEXT: matmul.mxu0 rd=0, rs1=2, rs2=1
