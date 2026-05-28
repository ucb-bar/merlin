#kb_target_llvm_cpu_spacemit_x60 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">

#pipeline_layout_saturnopu_add_f32 = #hal.pipeline.layout<constants = 0, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32 = #hal.pipeline.layout<constants = 9, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

#pipeline_layout_saturnopu_pooling_nchw_max_f32 = #hal.pipeline.layout<constants = 8, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

#pipeline_layout_saturnopu_bias_add_3d_f32 = #hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

#pipeline_layout_saturnopu_matmul_f32 = #hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

#pipeline_layout_saturnopu_linear_f32 = #hal.pipeline.layout<constants = 3, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

module attributes {transform.with_named_sequence} {

  hal.executable.source private @kb_saturnopu_add_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_add_f32.d5e4e21df82971a4.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @add_f32 ordinal(0)
        layout(#pipeline_layout_saturnopu_add_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @add_f32_workgroup(memref<?xf32>, memref<?xf32>, memref<?xf32>, index) attributes {hal.import.static}
        func.func @add_f32() {
          %c0 = arith.constant 0 : index
          %dim = hal.interface.workgroup.count[0] : index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32) binding(0) alignment(64) offset(%c0) : memref<?xf32>{%dim}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32) binding(1) alignment(64) offset(%c0) : memref<?xf32>{%dim}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32) binding(2) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          func.call @add_f32_workgroup(%binding0, %binding1, %binding2, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  hal.executable.source private @kb_saturnopu_conv_2d_nchw_fchw_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_conv_2d_nchw_fchw_f32.22499f1dfcb11636.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @conv_2d_nchw_fchw ordinal(0)
        layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @conv_2d_nchw_fchw_workgroup(memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index, index, index) attributes {hal.import.static}
        func.func @conv_2d_nchw_fchw() {
          %c0 = arith.constant 0 : index
          %N_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(0) : i32
          %N = arith.index_cast %N_i32 : i32 to index
          %C_in_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(1) : i32
          %C_in = arith.index_cast %C_in_i32 : i32 to index
          %H_in_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(2) : i32
          %H_in = arith.index_cast %H_in_i32 : i32 to index
          %W_in_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(3) : i32
          %W_in = arith.index_cast %W_in_i32 : i32 to index
          %F_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(4) : i32
          %F = arith.index_cast %F_i32 : i32 to index
          %KH_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(5) : i32
          %KH = arith.index_cast %KH_i32 : i32 to index
          %KW_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(6) : i32
          %KW = arith.index_cast %KW_i32 : i32 to index
          %H_out_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(7) : i32
          %H_out = arith.index_cast %H_out_i32 : i32 to index
          %W_out_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) ordinal(8) : i32
          %W_out = arith.index_cast %W_out_i32 : i32 to index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) binding(0) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %C_in, %H_in, %W_in}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) binding(1) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%F, %C_in, %KH, %KW}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) binding(2) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %F, %H_out, %W_out}
        %binding3 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_conv_2d_nchw_fchw_f32) binding(3) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %F, %H_out, %W_out}
          func.call @conv_2d_nchw_fchw_workgroup(%binding0, %binding1, %binding2, %binding3, %N, %C_in, %H_in, %W_in, %F, %KH, %KW, %H_out, %W_out, %tid) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index, index, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  hal.executable.source private @kb_saturnopu_pooling_nchw_max_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_pooling_nchw_max_f32.490327ae24ea2d19.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @pooling_nchw_max ordinal(0)
        layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @pooling_nchw_max_workgroup(memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index, index) attributes {hal.import.static}
        func.func @pooling_nchw_max() {
          %c0 = arith.constant 0 : index
          %N_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(0) : i32
          %N = arith.index_cast %N_i32 : i32 to index
          %C_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(1) : i32
          %C = arith.index_cast %C_i32 : i32 to index
          %H_in_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(2) : i32
          %H_in = arith.index_cast %H_in_i32 : i32 to index
          %W_in_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(3) : i32
          %W_in = arith.index_cast %W_in_i32 : i32 to index
          %KH_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(4) : i32
          %KH = arith.index_cast %KH_i32 : i32 to index
          %KW_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(5) : i32
          %KW = arith.index_cast %KW_i32 : i32 to index
          %H_out_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(6) : i32
          %H_out = arith.index_cast %H_out_i32 : i32 to index
          %W_out_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) ordinal(7) : i32
          %W_out = arith.index_cast %W_out_i32 : i32 to index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) binding(0) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %C, %H_in, %W_in}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) binding(1) alignment(64) offset(%c0) : memref<?x?xf32>{%KH, %KW}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) binding(2) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %C, %H_out, %W_out}
        %binding3 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_pooling_nchw_max_f32) binding(3) alignment(64) offset(%c0) : memref<?x?x?x?xf32>{%N, %C, %H_out, %W_out}
          func.call @pooling_nchw_max_workgroup(%binding0, %binding1, %binding2, %binding3, %N, %C, %H_in, %W_in, %KH, %KW, %H_out, %W_out, %tid) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  hal.executable.source private @kb_saturnopu_bias_add_3d_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_bias_add_3d_f32.5b30736507586f4b.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @bias_add_3d_f32 ordinal(0)
        layout(#pipeline_layout_saturnopu_bias_add_3d_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @bias_add_3d_f32_workgroup(memref<?x?x?xf32>, memref<?xf32>, memref<?x?x?xf32>, index, index, index, index) attributes {hal.import.static}
        func.func @bias_add_3d_f32() {
          %c0 = arith.constant 0 : index
          %C_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_bias_add_3d_f32) ordinal(0) : i32
          %C = arith.index_cast %C_i32 : i32 to index
          %H_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_bias_add_3d_f32) ordinal(1) : i32
          %H = arith.index_cast %H_i32 : i32 to index
          %W_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_bias_add_3d_f32) ordinal(2) : i32
          %W = arith.index_cast %W_i32 : i32 to index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_bias_add_3d_f32) binding(0) alignment(64) offset(%c0) : memref<?x?x?xf32>{%C, %H, %W}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_bias_add_3d_f32) binding(1) alignment(64) offset(%c0) : memref<?xf32>{%C}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_bias_add_3d_f32) binding(2) alignment(64) offset(%c0) : memref<?x?x?xf32>{%C, %H, %W}
          func.call @bias_add_3d_f32_workgroup(%binding0, %binding1, %binding2, %C, %H, %W, %tid) : (memref<?x?x?xf32>, memref<?xf32>, memref<?x?x?xf32>, index, index, index, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  hal.executable.source private @kb_saturnopu_matmul_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_matmul_f32.2715fbbd1e914847.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @matmul_f32 ordinal(0)
        layout(#pipeline_layout_saturnopu_matmul_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @matmul_f32_workgroup(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) attributes {hal.import.static}
        func.func @matmul_f32() {
          %c0 = arith.constant 0 : index
          %M_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_matmul_f32) ordinal(0) : i32
          %M = arith.index_cast %M_i32 : i32 to index
          %K_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_matmul_f32) ordinal(1) : i32
          %K = arith.index_cast %K_i32 : i32 to index
          %N_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_matmul_f32) ordinal(2) : i32
          %N = arith.index_cast %N_i32 : i32 to index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_matmul_f32) binding(0) alignment(64) offset(%c0) : memref<?x?xf32>{%M, %K}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_matmul_f32) binding(1) alignment(64) offset(%c0) : memref<?x?xf32>{%K, %N}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_matmul_f32) binding(2) alignment(64) offset(%c0) : memref<?x?xf32>{%M, %N}
          func.call @matmul_f32_workgroup(%binding0, %binding1, %binding2, %M, %K, %N, %tid) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  hal.executable.source private @kb_saturnopu_linear_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_linear_f32.c76abf6927a02c40.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @linear_f32 ordinal(0)
        layout(#pipeline_layout_saturnopu_linear_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
      builtin.module {
        func.func private @linear_f32_workgroup(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) attributes {hal.import.static}
        func.func @linear_f32() {
          %c0 = arith.constant 0 : index
          %M_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_linear_f32) ordinal(0) : i32
          %M = arith.index_cast %M_i32 : i32 to index
          %K_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_linear_f32) ordinal(1) : i32
          %K = arith.index_cast %K_i32 : i32 to index
          %N_i32 = hal.interface.constant.load layout(#pipeline_layout_saturnopu_linear_f32) ordinal(2) : i32
          %N = arith.index_cast %N_i32 : i32 to index
          %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_linear_f32) binding(0) alignment(64) offset(%c0) : memref<?x?xf32>{%M, %K}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_linear_f32) binding(1) alignment(64) offset(%c0) : memref<?x?xf32>{%N, %K}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_linear_f32) binding(2) alignment(64) offset(%c0) : memref<?x?xf32>{%M, %N}
          func.call @linear_f32_workgroup(%binding0, %binding1, %binding2, %M, %K, %N, %tid) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, index) -> ()
          return
        }
      }
  }  // hal.executable.source

  util.func private @call_saturnopu_add_f32(%in0: tensor<?xf32>, %in1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %in0, %c0 : tensor<?xf32>
    %0 = flow.dispatch @kb_saturnopu_add_f32::@add_f32[%dim](%in0, %in1) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
    util.return %0 : tensor<?xf32>
  }

  util.func private @call_saturnopu_conv_2d_nchw_fchw_f32(%in0: tensor<?x?x?x?xf32>, %in1: tensor<?x?x?x?xf32>, %in2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %c_axis_0 = arith.constant 0 : index
    %c_axis_1 = arith.constant 1 : index
    %c_axis_2 = arith.constant 2 : index
    %c_axis_3 = arith.constant 3 : index
    %N = tensor.dim %in0, %c_axis_0 : tensor<?x?x?x?xf32>
    %C_in = tensor.dim %in0, %c_axis_1 : tensor<?x?x?x?xf32>
    %H_in = tensor.dim %in0, %c_axis_2 : tensor<?x?x?x?xf32>
    %W_in = tensor.dim %in0, %c_axis_3 : tensor<?x?x?x?xf32>
    %F = tensor.dim %in1, %c_axis_0 : tensor<?x?x?x?xf32>
    %KH = tensor.dim %in1, %c_axis_2 : tensor<?x?x?x?xf32>
    %KW = tensor.dim %in1, %c_axis_3 : tensor<?x?x?x?xf32>
    %H_out = tensor.dim %in2, %c_axis_2 : tensor<?x?x?x?xf32>
    %W_out = tensor.dim %in2, %c_axis_3 : tensor<?x?x?x?xf32>
    %N_i32 = arith.index_cast %N : index to i32
    %C_in_i32 = arith.index_cast %C_in : index to i32
    %H_in_i32 = arith.index_cast %H_in : index to i32
    %W_in_i32 = arith.index_cast %W_in : index to i32
    %F_i32 = arith.index_cast %F : index to i32
    %KH_i32 = arith.index_cast %KH : index to i32
    %KW_i32 = arith.index_cast %KW : index to i32
    %H_out_i32 = arith.index_cast %H_out : index to i32
    %W_out_i32 = arith.index_cast %W_out : index to i32
    %workload_1 = arith.muli %N, %F : index
    %workload_2 = arith.muli %workload_1, %H_out : index
    %workload = arith.muli %workload_2, %W_out : index
    %0 = flow.dispatch @kb_saturnopu_conv_2d_nchw_fchw_f32::@conv_2d_nchw_fchw[%workload](%N_i32, %C_in_i32, %H_in_i32, %W_in_i32, %F_i32, %KH_i32, %KW_i32, %H_out_i32, %W_out_i32, %in0, %in1, %in2) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?x?x?xf32>{%N, %C_in, %H_in, %W_in}, tensor<?x?x?x?xf32>{%F, %C_in, %KH, %KW}, tensor<?x?x?x?xf32>{%N, %F, %H_out, %W_out}) -> tensor<?x?x?x?xf32>{%N, %F, %H_out, %W_out}
    util.return %0 : tensor<?x?x?x?xf32>
  }

  util.func private @call_saturnopu_pooling_nchw_max_f32(%in0: tensor<?x?x?x?xf32>, %in1: tensor<?x?xf32>, %in2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %c_axis_0 = arith.constant 0 : index
    %c_axis_1 = arith.constant 1 : index
    %c_axis_2 = arith.constant 2 : index
    %c_axis_3 = arith.constant 3 : index
    %N = tensor.dim %in0, %c_axis_0 : tensor<?x?x?x?xf32>
    %C = tensor.dim %in0, %c_axis_1 : tensor<?x?x?x?xf32>
    %H_in = tensor.dim %in0, %c_axis_2 : tensor<?x?x?x?xf32>
    %W_in = tensor.dim %in0, %c_axis_3 : tensor<?x?x?x?xf32>
    %KH = tensor.dim %in1, %c_axis_0 : tensor<?x?xf32>
    %KW = tensor.dim %in1, %c_axis_1 : tensor<?x?xf32>
    %H_out = tensor.dim %in2, %c_axis_2 : tensor<?x?x?x?xf32>
    %W_out = tensor.dim %in2, %c_axis_3 : tensor<?x?x?x?xf32>
    %N_i32 = arith.index_cast %N : index to i32
    %C_i32 = arith.index_cast %C : index to i32
    %H_in_i32 = arith.index_cast %H_in : index to i32
    %W_in_i32 = arith.index_cast %W_in : index to i32
    %KH_i32 = arith.index_cast %KH : index to i32
    %KW_i32 = arith.index_cast %KW : index to i32
    %H_out_i32 = arith.index_cast %H_out : index to i32
    %W_out_i32 = arith.index_cast %W_out : index to i32
    %workload_1 = arith.muli %N, %C : index
    %workload_2 = arith.muli %workload_1, %H_out : index
    %workload = arith.muli %workload_2, %W_out : index
    %0 = flow.dispatch @kb_saturnopu_pooling_nchw_max_f32::@pooling_nchw_max[%workload](%N_i32, %C_i32, %H_in_i32, %W_in_i32, %KH_i32, %KW_i32, %H_out_i32, %W_out_i32, %in0, %in1, %in2) : (i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?x?x?xf32>{%N, %C, %H_in, %W_in}, tensor<?x?xf32>{%KH, %KW}, tensor<?x?x?x?xf32>{%N, %C, %H_out, %W_out}) -> tensor<?x?x?x?xf32>{%N, %C, %H_out, %W_out}
    util.return %0 : tensor<?x?x?x?xf32>
  }

  util.func private @call_saturnopu_bias_add_3d_f32(%in0: tensor<?x?x?xf32>, %in1: tensor<?xf32>) -> tensor<?x?x?xf32> {
    %c_axis_0 = arith.constant 0 : index
    %c_axis_1 = arith.constant 1 : index
    %c_axis_2 = arith.constant 2 : index
    %C = tensor.dim %in0, %c_axis_0 : tensor<?x?x?xf32>
    %H = tensor.dim %in0, %c_axis_1 : tensor<?x?x?xf32>
    %W = tensor.dim %in0, %c_axis_2 : tensor<?x?x?xf32>
    %C_i32 = arith.index_cast %C : index to i32
    %H_i32 = arith.index_cast %H : index to i32
    %W_i32 = arith.index_cast %W : index to i32
    %workload_1 = arith.muli %C, %H : index
    %workload = arith.muli %workload_1, %W : index
    %0 = flow.dispatch @kb_saturnopu_bias_add_3d_f32::@bias_add_3d_f32[%workload](%C_i32, %H_i32, %W_i32, %in0, %in1) : (i32, i32, i32, tensor<?x?x?xf32>{%C, %H, %W}, tensor<?xf32>{%C}) -> tensor<?x?x?xf32>{%C, %H, %W}
    util.return %0 : tensor<?x?x?xf32>
  }

  util.func private @call_saturnopu_matmul_f32(%in0: tensor<?x?xf32>, %in1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c_axis_0 = arith.constant 0 : index
    %c_axis_1 = arith.constant 1 : index
    %M = tensor.dim %in0, %c_axis_0 : tensor<?x?xf32>
    %K = tensor.dim %in0, %c_axis_1 : tensor<?x?xf32>
    %N = tensor.dim %in1, %c_axis_1 : tensor<?x?xf32>
    %M_i32 = arith.index_cast %M : index to i32
    %K_i32 = arith.index_cast %K : index to i32
    %N_i32 = arith.index_cast %N : index to i32
    %workload = arith.muli %M, %N : index
    %0 = flow.dispatch @kb_saturnopu_matmul_f32::@matmul_f32[%workload](%M_i32, %K_i32, %N_i32, %in0, %in1) : (i32, i32, i32, tensor<?x?xf32>{%M, %K}, tensor<?x?xf32>{%K, %N}) -> tensor<?x?xf32>{%M, %N}
    util.return %0 : tensor<?x?xf32>
  }

  util.func private @call_saturnopu_linear_f32(%in0: tensor<?x?xf32>, %in1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c_axis_0 = arith.constant 0 : index
    %c_axis_1 = arith.constant 1 : index
    %M = tensor.dim %in0, %c_axis_0 : tensor<?x?xf32>
    %K = tensor.dim %in0, %c_axis_1 : tensor<?x?xf32>
    %N = tensor.dim %in1, %c_axis_0 : tensor<?x?xf32>
    %M_i32 = arith.index_cast %M : index to i32
    %K_i32 = arith.index_cast %K : index to i32
    %N_i32 = arith.index_cast %N : index to i32
    %workload = arith.muli %M, %N : index
    %0 = flow.dispatch @kb_saturnopu_linear_f32::@linear_f32[%workload](%M_i32, %K_i32, %N_i32, %in0, %in1) : (i32, i32, i32, tensor<?x?xf32>{%M, %K}, tensor<?x?xf32>{%N, %K}) -> tensor<?x?xf32>{%M, %N}
    util.return %0 : tensor<?x?xf32>
  }

  transform.named_sequence @match_saturnopu_add_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
// Matches a 1-D linalg.generic f32 elementwise add with dynamic shape. The
// auto-generated cast_and_call sequence inserts tensor.cast to bridge to
// statically-shaped payload (see
// transform.type_conversion.tensor.cast_shape_dynamic_dims).

^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
  %add = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_add_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_add_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_add_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_saturnopu_conv_2d_nchw_fchw_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
^bb0(%in0: tensor<?x?x?x?xf32>, %in1: tensor<?x?x?x?xf32>, %in2: tensor<?x?x?x?xf32>):
  %op = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%in0, %in1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%in2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_conv_2d_nchw_fchw_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_conv_2d_nchw_fchw_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_conv_2d_nchw_fchw_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_saturnopu_pooling_nchw_max_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
^bb0(%in0: tensor<?x?x?x?xf32>, %in1: tensor<?x?xf32>, %in2: tensor<?x?x?x?xf32>):
  %op = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%in0, %in1 : tensor<?x?x?x?xf32>, tensor<?x?xf32>) outs(%in2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_pooling_nchw_max_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_pooling_nchw_max_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_pooling_nchw_max_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_saturnopu_bias_add_3d_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
// Matches dronet's per-conv bias-add pattern:
//   out[c, h, w] = in[c, h, w] + bias[c]
// Indexing maps:
//   #map3 = (d0, d1, d2) -> (d0, d1, d2)
//   #map4 = (d0, d1, d2) -> (d0)

^bb0(%in: tensor<?x?x?xf32>, %bias: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %in, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %in, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %in, %c2 : tensor<?x?x?xf32>
  %empty = tensor.empty(%dim0, %dim1, %dim2) {"match.operation_name_only"} : tensor<?x?x?xf32>
  %add = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%in, %bias : tensor<?x?x?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?x?x?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?x?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_bias_add_3d_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_bias_add_3d_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_bias_add_3d_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_saturnopu_matmul_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
// Matches the standard `linalg.matmul` named op over f32 with untransposed
// RHS — out[m, n] = sum_k lhs[m, k] * rhs[k, n]. This is the form dronet's
// final classifier dot ends up in after im2col preprocessing (and with
// data tiling disabled so the encoding annotations don't appear).
//
// Body uses `linalg.matmul` directly (not the generic form) because IREE's
// preprocessing keeps it as a named op when standard layout matches.

^bb0(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %n = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) {"match.operation_name_only"} : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_matmul_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_matmul_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_matmul_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_saturnopu_linear_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
// Matches a 2D f32 matmul with transposed B (rhs interpreted as (N, K)
// row-major). Expressed as a linalg.generic with explicit indexing maps
// rather than `linalg.matmul_transpose_b` because not every in-tree IREE
// version registers that named op. Equivalent semantics:
//   out[m, n] = sum_k lhs[m, k] * rhs[n, k]
// Matches the convention `kernel_linear` uses in `../rvv_linear_direct.c`:
// `weight + n * K` to scan a column of the conceptual matmul.

^bb0(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>):
  %c0 = arith.constant 0 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) {"match.operation_name_only"} : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%filled : tensor<?x?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
  } -> tensor<?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @cast_and_call_saturnopu_linear_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_linear_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_linear_f32 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
        transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
    ^bb1(%f: !transform.any_op):
      transform.foreach_match in %f
        @match_saturnopu_add_f32 -> @cast_and_call_saturnopu_add_f32,
        @match_saturnopu_conv_2d_nchw_fchw_f32 -> @cast_and_call_saturnopu_conv_2d_nchw_fchw_f32,
        @match_saturnopu_pooling_nchw_max_f32 -> @cast_and_call_saturnopu_pooling_nchw_max_f32,
        @match_saturnopu_bias_add_3d_f32 -> @cast_and_call_saturnopu_bias_add_3d_f32,
        @match_saturnopu_matmul_f32 -> @cast_and_call_saturnopu_matmul_f32,
        @match_saturnopu_linear_f32 -> @cast_and_call_saturnopu_linear_f32
        : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }

}
