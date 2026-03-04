hal.executable public @matmul_dispatch_0 {
  hal.executable.variant public @embedded_elf_riscv_64 target(<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+buddyext", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @matmul_dispatch_0_matmul_4x4x4_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_dispatch_0_matmul_4x4x4_f32() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>> -> tensor<4x4xf32>
        %6 = linalg.matmul ins(%3, %4 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%5 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4xf32>>
        return
      }
    }
  }
}
