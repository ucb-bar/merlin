"builtin.module"() ({
  "util.global"() <{initial_value = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]> : !hal.device, sym_name = "__device_0", sym_visibility = "private", type = !hal.device}> : () -> ()
  "util.func"() <{function_type = (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view, sym_name = "main"}> ({
  ^bb0(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view):
    %0 = "hal.tensor.import"(%arg0) <{name = "input0", operandSegmentSizes = array<i32: 1, 0, 0>, target_encoding = tensor<16x1024xf8E4M3FN>}> : (!hal.buffer_view) -> tensor<16x1024xf8E4M3FN>
    %1 = "iree_tensor_ext.compute_barrier.start"(%0) : (tensor<16x1024xf8E4M3FN>) -> tensor<16x1024xf8E4M3FN>
    %2 = "hal.tensor.import"(%arg1) <{name = "input1", operandSegmentSizes = array<i32: 1, 0, 0>, target_encoding = tensor<1024x128xf8E4M3FN>}> : (!hal.buffer_view) -> tensor<1024x128xf8E4M3FN>
    %3 = "iree_tensor_ext.compute_barrier.start"(%2) : (tensor<1024x128xf8E4M3FN>) -> tensor<1024x128xf8E4M3FN>
    "npu_isa.dma_load"() <{base = 0 : i64, flag = 0 : i64, rd = 2 : i64, size = 2048 : i64}> : () -> ()
    "npu_isa.dma_wait"() <{flag = 0 : i64}> : () -> ()
    "npu_isa.dma_load_mxu0"() <{base = 2048 : i64, flag = 0 : i64, rd = 1 : i64, size = 512 : i64}> : () -> ()
    "npu_isa.dma_wait"() <{flag = 0 : i64}> : () -> ()
    %4 = "npu_isa.matmul_mxu0"(%1, %3) <{rd = 0 : i64, rs1 = 2 : i64, rs2 = 1 : i64}> : (tensor<16x1024xf8E4M3FN>, tensor<1024x128xf8E4M3FN>) -> tensor<16x128xf32>
    %5 = "iree_tensor_ext.compute_barrier.end"(%4) : (tensor<16x128xf32>) -> tensor<16x128xf32>
    %6 = "hal.tensor.export"(%5) <{name = "output0", source_encoding = tensor<16x128xf32>}> : (tensor<16x128xf32>) -> !hal.buffer_view
    "util.return"(%6) : (!hal.buffer_view) -> ()
  }) {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<16x1024xf8E4M3FN>, %input1: tensor<1024x128xf8E4M3FN>) -> (%output0: tensor<16x128xf32>)"}} : () -> ()
}) {stream.affinity.default = #hal.device.affinity<@__device_0>} : () -> ()
