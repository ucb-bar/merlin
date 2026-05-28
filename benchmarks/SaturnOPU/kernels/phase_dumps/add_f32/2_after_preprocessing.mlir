#executable_target_embedded_elf_riscv_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">
#executable_target_embedded_elf_riscv_64_1 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, link_ukernel_bitcode = false, loop_vectorization = true, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_riscv_64_1]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.func private @call_saturnopu_add_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %0 = flow.dispatch @kb_saturnopu_add_f32::@add_f32[%dim](%arg0, %arg1) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
    util.return %0 : tensor<?xf32>
  }
  hal.executable.source private @kb_saturnopu_add_f32 attributes {objects = #hal.executable.objects<{#executable_target_embedded_elf_riscv_64 = [#hal.executable.object<{path = "saturnopu_add_f32.3bf893a6d973aaf1.riscv64-none-elf.o"}>]}>} {
    hal.executable.export public @add_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %arg1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func private @add_f32_workgroup(memref<?xf32>, memref<?xf32>, memref<?xf32>, index) attributes {hal.import.static}
      func.func @add_f32() {
        %c0 = arith.constant 0 : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<?xf32>{%workgroup_count_x}
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<?xf32>{%workgroup_count_x}
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<?xf32>{%workgroup_count_x}
        call @add_f32_workgroup(%0, %1, %2, %workgroup_id_x) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
        return
      }
    }
  }
  util.global private @__device_0 = #device_target_local
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @main(%input0: tensor<8xf32>, %input1: tensor<8xf32>) -> (%output0: tensor<8xf32>)", iree.abi.model = "coarse-fences"}} {
    %0 = hal.tensor.import wait(%arg2) => %arg0 "input0" : !hal.buffer_view -> tensor<8xf32>
    %1 = hal.tensor.import wait(%arg2) => %arg1 "input1" : !hal.buffer_view -> tensor<8xf32>
    %cast = tensor.cast %0 : tensor<8xf32> to tensor<?xf32>
    %cast_0 = tensor.cast %1 : tensor<8xf32> to tensor<?xf32>
    %2 = util.call @call_saturnopu_add_f32(%cast, %cast_0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %cast_1 = tensor.cast %2 : tensor<?xf32> to tensor<8xf32>
    %3 = hal.tensor.barrier join(%cast_1 : tensor<8xf32>) => %arg3 : !hal.fence
    %4 = hal.tensor.export %3 "output0" : tensor<8xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}
