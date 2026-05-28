#executable_target_embedded_elf_riscv_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">
#executable_target_embedded_elf_riscv_64_1 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, link_ukernel_bitcode = false, loop_vectorization = true, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer>]>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_riscv_64_1]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
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
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c8]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<8xf32> in !stream.resource<external>{%c32}
    %1 = stream.timepoint.import on(#hal.device.affinity<@__device_0>) %arg2 : (!hal.fence) => !stream.timepoint
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c8]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<8xf32> in !stream.resource<external>{%c32}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%1) => !stream.resource<external>{%c32} => !stream.timepoint
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%1) => !stream.resource<transient>{%c128} => !stream.timepoint
    %3 = stream.timepoint.join max(%result_timepoint, %result_timepoint_1) => !stream.timepoint
    %4 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%3) => with(%0 as %arg4: !stream.resource<external>{%c32}, %2 as %arg5: !stream.resource<external>{%c32}, %result as %arg6: !stream.resource<external>{%c32}, %result_0 as %arg7: !stream.resource<transient>{%c128}) {
      stream.cmd.concurrent {
        stream.cmd.copy %arg4[%c0], %arg7[%c0], %c32 : !stream.resource<external>{%c32} -> !stream.resource<transient>{%c128}
        stream.cmd.copy %arg5[%c0], %arg7[%c64], %c32 : !stream.resource<external>{%c32} -> !stream.resource<transient>{%c128}
      }
      stream.cmd.dispatch @kb_saturnopu_add_f32::@add_f32[%c8] {
        ro %arg7[%c0 for %c32] : !stream.resource<transient>{%c128},
        ro %arg7[%c64 for %c32] : !stream.resource<transient>{%c128},
        wo %arg6[%c0 for %c32] : !stream.resource<external>{%c32}
      }
    } => !stream.timepoint
    %5 = stream.resource.dealloca on(#hal.device.affinity<@__device_0>) await(%4) => %result_0 : !stream.resource<transient>{%c128} => !stream.timepoint
    stream.timepoint.chain_external on(#hal.device.affinity<@__device_0>) %5 => (%arg3 : !hal.fence)
    %6 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %result : tensor<8xf32> in !stream.resource<external>{%c32} -> !hal.buffer_view
    util.return %6 : !hal.buffer_view
  }
}
