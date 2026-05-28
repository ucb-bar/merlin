#executable_target_embedded_elf_riscv_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">
#executable_target_embedded_elf_riscv_64_1 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, link_ukernel_bitcode = false, loop_vectorization = true, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_riscv_64_1]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.func private @call_saturnopu_linear_f32(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %0 = arith.index_cast %dim : index to i32
    %1 = arith.index_cast %dim_0 : index to i32
    %2 = arith.index_cast %dim_1 : index to i32
    %3 = arith.muli %dim, %dim_1 : index
    %4 = flow.dispatch @kb_saturnopu_linear_f32::@linear_f32[%3](%0, %1, %2, %arg0, %arg1) : (i32, i32, i32, tensor<?x?xf32>{%dim, %dim_0}, tensor<?x?xf32>{%dim_1, %dim_0}) -> tensor<?x?xf32>{%dim, %dim_1}
    util.return %4 : tensor<?x?xf32>
  }
  hal.executable.source private @kb_saturnopu_linear_f32 attributes {objects = #hal.executable.objects<{#executable_target_embedded_elf_riscv_64 = [#hal.executable.object<{path = "saturnopu_linear_f32.9f1e722354a3a5b8.riscv64-none-elf.o"}>]}>} {
    hal.executable.export public @linear_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %arg1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func private @linear_f32_workgroup(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) attributes {hal.import.static}
      func.func @linear_f32() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
        %5 = arith.index_cast %4 : i32 to index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<?x?xf32>{%1, %3}
        %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<?x?xf32>{%5, %3}
        %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<?x?xf32>{%1, %5}
        call @linear_f32_workgroup(%6, %7, %8, %1, %3, %5, %workgroup_id_x) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) -> ()
        return
      }
    }
  }
  util.global private @__device_0 = #device_target_local
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @main(%input0: tensor<32x32xf32>, %input1: tensor<32x32xf32>) -> (%output0: tensor<32x32xf32>)", iree.abi.model = "coarse-fences"}} {
    %0 = hal.tensor.import wait(%arg2) => %arg0 "input0" : !hal.buffer_view -> tensor<32x32xf32>
    %1 = hal.tensor.import wait(%arg2) => %arg1 "input1" : !hal.buffer_view -> tensor<32x32xf32>
    %cast = tensor.cast %0 : tensor<32x32xf32> to tensor<?x?xf32>
    %cast_0 = tensor.cast %1 : tensor<32x32xf32> to tensor<?x?xf32>
    %2 = util.call @call_saturnopu_linear_f32(%cast, %cast_0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %cast_1 = tensor.cast %2 : tensor<?x?xf32> to tensor<32x32xf32>
    %3 = hal.tensor.barrier join(%cast_1 : tensor<32x32xf32>) => %arg3 : !hal.fence
    %4 = hal.tensor.export %3 "output0" : tensor<32x32xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}
