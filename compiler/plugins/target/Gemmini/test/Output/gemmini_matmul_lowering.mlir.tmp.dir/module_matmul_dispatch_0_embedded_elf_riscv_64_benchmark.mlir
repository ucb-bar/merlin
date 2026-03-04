module {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+buddyext", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf"}>]> : !hal.device
  hal.executable private @matmul_dispatch_0 {
    hal.executable.variant public @embedded_elf_riscv_64 target(<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+buddyext", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf"}>) {
      hal.executable.export public @matmul_dispatch_0_matmul_4x4x4_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @matmul_dispatch_0_matmul_4x4x4_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {enable_loop_peeling}>} {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
          %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
          %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>
          %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
          %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
          %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>> -> tensor<4x4xf32>
          %6 = linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [4, 4, 0], distribution = [4, 4, 0], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>} ins(%3, %4 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%5 : tensor<4x4xf32>) -> tensor<4x4xf32>
          iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>
          return
        }
      }
    }
  }
  util.global private mutable @matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer
  util.initializer {
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
    %c768 = arith.constant 768 : index
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%c768}
    util.global.store %buffer, @matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer
    util.return
  }
  util.func public @matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %0 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer = util.global.load @matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@matmul_dispatch_0::@embedded_elf_riscv_64::@matmul_dispatch_0_matmul_4x4x4_f32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@matmul_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@matmul_dispatch_0::@embedded_elf_riscv_64::@matmul_dispatch_0_matmul_4x4x4_f32) : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
        (%matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer)[%c0, %c64], 
        (%matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer)[%c256, %c64], 
        (%matmul_dispatch_0_embedded_elf_riscv_64_matmul_dispatch_0_matmul_4x4x4_f32_buffer : !hal.buffer)[%c512, %c64]
      ]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%1) signal(%fence) commands(%cmd) flags("None")
    %c-1_i32 = arith.constant -1 : i32
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
