#executable_target_embedded_elf_riscv_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu = "", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b,+zfh,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, loop_vectorization = true, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_abi = "lp64d", target_triple = "riscv64-unknown-unknown-eabi-elf", ukernels = "all"}>
#map = affine_map<(d0, d1, d2) -> (d2, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d0)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 + d3, d1 + d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d1)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0 * 2 + d3, d1 * 2 + d4)>
#map18 = affine_map<(d0) -> (d0)>
#device_target_primary = #hal.device.target<"local", [#executable_target_embedded_elf_riscv_64]> : !hal.device
#device_target_secondary = #hal.device.target<"local", [#executable_target_embedded_elf_riscv_64]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 12544, 32]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 96]>
#encoding2 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 144]>
#encoding3 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 144]>
#encoding4 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 192]>
#encoding5 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 192]>
#encoding6 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 384]>
#encoding7 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 384]>
#encoding8 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 576]>
#encoding9 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 576]>
#encoding10 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 960]>
#encoding11 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [320, 49, 960]>
#encoding12 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1280, 49, 320]>
#encoding13 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map4, #map5, #map3], iteration_sizes = [1000, 1280]>
#encoding14 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map4, #map5, #map3], iteration_sizes = [1000, 1280]>
#encoding15 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 12544, 32]>
#encoding16 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [16, 12544, 32]>
#encoding17 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 96]>
#encoding18 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 96]>
#encoding19 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 144]>
#encoding20 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [24, 3136, 144]>
#encoding21 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 144]>
#encoding22 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 144]>
#encoding23 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 192]>
#encoding24 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 784, 192]>
#encoding25 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 192]>
#encoding26 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 192]>
#encoding27 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 384]>
#encoding28 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [64, 196, 384]>
#encoding29 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 384]>
#encoding30 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 384]>
#encoding31 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 576]>
#encoding32 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [96, 196, 576]>
#encoding33 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 576]>
#encoding34 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 576]>
#encoding35 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 960]>
#encoding36 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [160, 49, 960]>
#encoding37 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [320, 49, 960]>
#encoding38 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1280, 49, 320]>
#encoding39 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [320, 49, 960]>
#encoding40 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [1280, 49, 320]>
#encoding41 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map4, #map5, #map3], iteration_sizes = [1000, 1280]>
#encoding42 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [16, 12544, 32]>
#encoding43 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [24, 3136, 96]>
#encoding44 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [24, 3136, 144]>
#encoding45 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [32, 784, 144]>
#encoding46 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [32, 784, 192]>
#encoding47 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [64, 196, 192]>
#encoding48 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [64, 196, 384]>
#encoding49 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [96, 196, 384]>
#encoding50 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [96, 196, 576]>
#encoding51 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [160, 49, 576]>
#encoding52 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [160, 49, 960]>
#encoding53 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, [#map2, #map3]], iteration_sizes = [320, 49, 960]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_primary : !hal.device
  util.global private @device_b = #device_target_secondary : !hal.device
  util.global private @__hoisted_tensor_16x32xf32__encoded {stream.affinity.default = #hal.device.affinity<@device_a>} : tensor<16x32xf32, #encoding>
  util.initializer attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
    %__constant_tensor_16x32xf32 = util.global.load immutable @__constant_tensor_16x32xf32 : tensor<16x32xf32>
    %0 = flow.tensor.encode %__constant_tensor_16x32xf32 : tensor<16x32xf32> -> tensor<16x32xf32, #encoding>
    util.global.store %0, @__hoisted_tensor_16x32xf32__encoded : tensor<16x32xf32, #encoding>
    util.return
  }
  util.global private @__hoisted_tensor_16xf32__encoded {stream.affinity.default = #hal.device.affinity<@device_a>} : tensor<16xf32, #encoding42>
  util.global private @__constant_tensor_16xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<[-2.14129257, 0.427362442, -1.28207529, 1.03522277, -0.101213448, -0.462517053, -0.66123563, -1.79533494, -0.838829994, -1.65666711, 1.61357224, -0.426707566, -0.719091117, -0.171038985, 1.63712561, 1.41761291]> : tensor<16xf32>
  util.initializer attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
    %__constant_tensor_16xf32 = util.global.load immutable @__constant_tensor_16xf32 : tensor<16xf32>
    %0 = flow.tensor.encode %__constant_tensor_16xf32 : tensor<16xf32> -> tensor<16xf32, #encoding42>
    util.global.store %0, @__hoisted_tensor_16xf32__encoded : tensor<16xf32, #encoding42>
    util.return
  }
  ...
  flow.executable private @torch_jit$async_dispatch_0 {
    flow.executable.export public @torch_jit$async_dispatch_0_slow_memcpy workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_0_slow_memcpy(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x224x224xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x226x226xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [3, 224, 224], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x224x224xf32>> -> tensor<3x224x224xf32>
        iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 1, 1], sizes = [3, 224, 224], strides = [1, 1, 1] : tensor<3x224x224xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x226x226xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_1 {
    flow.executable.export public @torch_jit$async_dispatch_1_conv_32x112x112x3x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_1_conv_32x112x112x3x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x226x226xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x114x114xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %cst_1 = arith.constant dense<[-8.599730e-02, 0.560411274, 0.348930985, 0.28295204, 0.967753171, 0.652543604, 0.49507159, 0.568947434, 0.617724597, 1.1074674E-4, -0.335959584, 0.958215057, 0.446063846, -0.371221632, -3.36437428E-4, 0.00633317837, -0.0415116511, -0.0181182344, 0.34100008, 0.100486755, -0.264608026, 0.497580558, 0.413815796, -0.0154052218, -0.48738268, 0.510827184, -0.302491188, 0.639285624, -0.121834084, -0.0693915486, 0.3928698, 0.270579666]> : tensor<32xf32>
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [3, 226, 226], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x226x226xf32>> -> tensor<3x226x226xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [32, 3, 3, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x3x3xf32>> -> tensor<32x3x3x3xf32>
        %2 = tensor.empty() : tensor<32x112x112xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x112x112xf32>) -> tensor<32x112x112xf32>
        %4 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<3x226x226xf32>, tensor<32x3x3x3xf32>) outs(%3 : tensor<32x112x112xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %6 = arith.mulf %in, %in_2 : f32
          %7 = arith.addf %out, %6 : f32
          linalg.yield %7 : f32
        } -> tensor<32x112x112xf32>
        %5 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %cst_1 : tensor<32x112x112xf32>, tensor<32xf32>) outs(%2 : tensor<32x112x112xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %6 = arith.addf %in, %in_2 : f32
          %7 = arith.cmpf ult, %6, %cst : f32
          %8 = arith.select %7, %cst, %6 : f32
          %9 = arith.cmpf ugt, %8, %cst_0 : f32
          %10 = arith.select %9, %cst_0, %8 : f32
          linalg.yield %10 : f32
        } -> tensor<32x112x112xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 1, 1], sizes = [32, 112, 112], strides = [1, 1, 1] : tensor<32x112x112xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x114x114xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_2 {
    flow.executable.export public @torch_jit$async_dispatch_2_conv_112x112x32x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_2_conv_112x112x32x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x114x114xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x112x112xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %cst_1 = arith.constant dense<[-0.0118963569, 0.93304044, 0.00959587097, 0.418227226, 0.261094362, 0.837611079, 0.873585463, 0.409204483, 3.73148608, -0.00466901483, 0.333401263, 0.326424628, 0.808532595, -0.412330955, -0.00567122735, -0.00552821811, 8.330520e-03, 0.0136085181, 0.00787018239, 0.0779734626, -0.287649542, 1.56298876, 0.878986835, 0.0012245276, 0.75949639, -0.280076623, -0.365501821, 0.879585862, 0.673719525, 0.321320176, 0.970255672, -0.202976912]> : tensor<32xf32>
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 114, 114], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x114x114xf32>> -> tensor<32x114x114xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [32, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x3xf32>> -> tensor<32x3x3xf32>
        %2 = tensor.empty() : tensor<32x112x112xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x112x112xf32>) -> tensor<32x112x112xf32>
        %4 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<32x114x114xf32>, tensor<32x3x3xf32>) outs(%3 : tensor<32x112x112xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %6 = arith.mulf %in, %in_2 : f32
          %7 = arith.addf %out, %6 : f32
          linalg.yield %7 : f32
        } -> tensor<32x112x112xf32>
        %5 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %cst_1 : tensor<32x112x112xf32>, tensor<32xf32>) outs(%2 : tensor<32x112x112xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %6 = arith.addf %in, %in_2 : f32
          %7 = arith.cmpf ult, %6, %cst : f32
          %8 = arith.select %7, %cst, %6 : f32
          %9 = arith.cmpf ugt, %8, %cst_0 : f32
          %10 = arith.select %9, %cst_0, %8 : f32
          linalg.yield %10 : f32
        } -> tensor<32x112x112xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 0, 0], sizes = [32, 112, 112], strides = [1, 1, 1] : tensor<32x112x112xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x112x112xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_3 {
    flow.executable.export public @torch_jit$async_dispatch_3_matmul_like_16x12544x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_3_matmul_like_16x12544x32_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x12544xf32, #encoding15>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x32xf32, #encoding>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding42>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x12544xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 12544], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x12544xf32, #encoding15>> -> tensor<32x12544xf32, #encoding15>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x32xf32, #encoding>> -> tensor<16x32xf32, #encoding>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding42>> -> tensor<16xf32, #encoding42>
        %3 = tensor.empty() : tensor<16x12544xf32, #encoding16>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<16x12544xf32, #encoding16>) -> tensor<16x12544xf32, #encoding16>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<32x12544xf32, #encoding15>, tensor<16x32xf32, #encoding>) outs(%4 : tensor<16x12544xf32, #encoding16>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<16x12544xf32, #encoding16>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<16x12544xf32, #encoding16>, tensor<16xf32, #encoding42>) outs(%3 : tensor<16x12544xf32, #encoding16>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<16x12544xf32, #encoding16>
        %7 = iree_encoding.unset_encoding %6 : tensor<16x12544xf32, #encoding16> -> tensor<16x12544xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [16, 12544], strides = [1, 1] : tensor<16x12544xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x12544xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_4 {
    flow.executable.export public @torch_jit$async_dispatch_4_matmul_like_96x112x112x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_4_matmul_like_96x112x112x16_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x112x112xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x16xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<96x114x114xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 112, 112], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x112x112xf32>> -> tensor<16x112x112xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [96, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x16xf32>> -> tensor<96x16xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [96], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32>> -> tensor<96xf32>
        %3 = tensor.empty() : tensor<96x112x112xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<96x112x112xf32>) -> tensor<96x112x112xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<16x112x112xf32>, tensor<96x16xf32>) outs(%4 : tensor<96x112x112xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<96x112x112xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<96x112x112xf32>, tensor<96xf32>) outs(%3 : tensor<96x112x112xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<96x112x112xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [96, 112, 112], strides = [1, 1, 1] : tensor<96x112x112xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<96x114x114xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_5 {
    flow.executable.export public @torch_jit$async_dispatch_5_conv_56x56x96x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_5_conv_56x56x96x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x114x114xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x3136xf32, #encoding17>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [96, 114, 114], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x114x114xf32>> -> tensor<96x114x114xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [96, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x3x3xf32>> -> tensor<96x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [96], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32>> -> tensor<96xf32>
        %3 = tensor.empty() : tensor<96x56x56xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<96x56x56xf32>) -> tensor<96x56x56xf32>
        %5 = linalg.generic {indexing_maps = [#map17, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<96x114x114xf32>, tensor<96x3x3xf32>) outs(%4 : tensor<96x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<96x56x56xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<96x56x56xf32>, tensor<96xf32>) outs(%3 : tensor<96x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<96x56x56xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<96x56x56xf32> into tensor<96x3136xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<96x3136xf32> -> tensor<96x3136xf32, #encoding17>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [96, 3136], strides = [1, 1] : tensor<96x3136xf32, #encoding17> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x3136xf32, #encoding17>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_6 {
    flow.executable.export public @torch_jit$async_dispatch_6_matmul_like_24x3136x96_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_6_matmul_like_24x3136x96_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x3136xf32, #encoding17>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x96xf32, #encoding1>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24xf32, #encoding43>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x3136xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [96, 3136], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x3136xf32, #encoding17>> -> tensor<96x3136xf32, #encoding17>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [24, 96], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x96xf32, #encoding1>> -> tensor<24x96xf32, #encoding1>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [24], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24xf32, #encoding43>> -> tensor<24xf32, #encoding43>
        %3 = tensor.empty() : tensor<24x3136xf32, #encoding18>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<24x3136xf32, #encoding18>) -> tensor<24x3136xf32, #encoding18>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<96x3136xf32, #encoding17>, tensor<24x96xf32, #encoding1>) outs(%4 : tensor<24x3136xf32, #encoding18>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<24x3136xf32, #encoding18>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<24x3136xf32, #encoding18>, tensor<24xf32, #encoding43>) outs(%3 : tensor<24x3136xf32, #encoding18>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<24x3136xf32, #encoding18>
        %7 = iree_encoding.unset_encoding %6 : tensor<24x3136xf32, #encoding18> -> tensor<24x3136xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [24, 3136], strides = [1, 1] : tensor<24x3136xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x3136xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_7 {
    flow.executable.export public @torch_jit$async_dispatch_7_matmul_like_144x56x56x24_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_7_matmul_like_144x56x56x24_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x56x56xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x24xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<144x58x58xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [24, 56, 56], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x56x56xf32>> -> tensor<24x56x56xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [144, 24], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x24xf32>> -> tensor<144x24xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [144], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>> -> tensor<144xf32>
        %3 = tensor.empty() : tensor<144x56x56xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<144x56x56xf32>) -> tensor<144x56x56xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<24x56x56xf32>, tensor<144x24xf32>) outs(%4 : tensor<144x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<144x56x56xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<144x56x56xf32>, tensor<144xf32>) outs(%3 : tensor<144x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<144x56x56xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [144, 56, 56], strides = [1, 1, 1] : tensor<144x56x56xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<144x58x58xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_8 {
    flow.executable.export public @torch_jit$async_dispatch_8_conv_56x56x144x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_8_conv_56x56x144x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x58x58xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<144x3136xf32, #encoding19>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [144, 58, 58], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x58x58xf32>> -> tensor<144x58x58xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [144, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3x3xf32>> -> tensor<144x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [144], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>> -> tensor<144xf32>
        %3 = tensor.empty() : tensor<144x56x56xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<144x56x56xf32>) -> tensor<144x56x56xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<144x58x58xf32>, tensor<144x3x3xf32>) outs(%4 : tensor<144x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<144x56x56xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<144x56x56xf32>, tensor<144xf32>) outs(%3 : tensor<144x56x56xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<144x56x56xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<144x56x56xf32> into tensor<144x3136xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<144x3136xf32> -> tensor<144x3136xf32, #encoding19>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [144, 3136], strides = [1, 1] : tensor<144x3136xf32, #encoding19> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<144x3136xf32, #encoding19>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_9 {
    flow.executable.export public @torch_jit$async_dispatch_9_matmul_like_24x3136x144_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_9_matmul_like_24x3136x144_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3136xf32, #encoding19>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x144xf32, #encoding2>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x3136xf32, #encoding20>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24xf32, #encoding44>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x3136xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [144, 3136], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3136xf32, #encoding19>> -> tensor<144x3136xf32, #encoding19>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [24, 144], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x144xf32, #encoding2>> -> tensor<24x144xf32, #encoding2>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [24, 3136], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x3136xf32, #encoding20>> -> tensor<24x3136xf32, #encoding20>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [24], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24xf32, #encoding44>> -> tensor<24xf32, #encoding44>
        %4 = tensor.empty() : tensor<24x3136xf32, #encoding20>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<24x3136xf32, #encoding20>) -> tensor<24x3136xf32, #encoding20>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<144x3136xf32, #encoding19>, tensor<24x144xf32, #encoding2>) outs(%5 : tensor<24x3136xf32, #encoding20>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<24x3136xf32, #encoding20>
        %7 = linalg.generic {indexing_maps = [#map5, #map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%2, %6, %3 : tensor<24x3136xf32, #encoding20>, tensor<24x3136xf32, #encoding20>, tensor<24xf32, #encoding44>) outs(%4 : tensor<24x3136xf32, #encoding20>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<24x3136xf32, #encoding20>
        %8 = iree_encoding.unset_encoding %7 : tensor<24x3136xf32, #encoding20> -> tensor<24x3136xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %arg4, offsets = [0, 0], sizes = [24, 3136], strides = [1, 1] : tensor<24x3136xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x3136xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_11 {
    flow.executable.export public @torch_jit$async_dispatch_11_conv_28x28x144x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_11_conv_28x28x144x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x58x58xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<144x784xf32, #encoding21>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [144, 58, 58], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x58x58xf32>> -> tensor<144x58x58xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [144, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x3x3xf32>> -> tensor<144x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [144], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144xf32>> -> tensor<144xf32>
        %3 = tensor.empty() : tensor<144x28x28xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<144x28x28xf32>) -> tensor<144x28x28xf32>
        %5 = linalg.generic {indexing_maps = [#map17, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<144x58x58xf32>, tensor<144x3x3xf32>) outs(%4 : tensor<144x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<144x28x28xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<144x28x28xf32>, tensor<144xf32>) outs(%3 : tensor<144x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<144x28x28xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<144x28x28xf32> into tensor<144x784xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<144x784xf32> -> tensor<144x784xf32, #encoding21>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [144, 784], strides = [1, 1] : tensor<144x784xf32, #encoding21> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<144x784xf32, #encoding21>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_12 {
    flow.executable.export public @torch_jit$async_dispatch_12_matmul_like_32x784x144_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_12_matmul_like_32x784x144_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x784xf32, #encoding21>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x144xf32, #encoding3>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding45>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x784xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [144, 784], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<144x784xf32, #encoding21>> -> tensor<144x784xf32, #encoding21>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [32, 144], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x144xf32, #encoding3>> -> tensor<32x144xf32, #encoding3>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [32], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding45>> -> tensor<32xf32, #encoding45>
        %3 = tensor.empty() : tensor<32x784xf32, #encoding22>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<32x784xf32, #encoding22>) -> tensor<32x784xf32, #encoding22>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<144x784xf32, #encoding21>, tensor<32x144xf32, #encoding3>) outs(%4 : tensor<32x784xf32, #encoding22>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<32x784xf32, #encoding22>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<32x784xf32, #encoding22>, tensor<32xf32, #encoding45>) outs(%3 : tensor<32x784xf32, #encoding22>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<32x784xf32, #encoding22>
        %7 = iree_encoding.unset_encoding %6 : tensor<32x784xf32, #encoding22> -> tensor<32x784xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [32, 784], strides = [1, 1] : tensor<32x784xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x784xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_13 {
    flow.executable.export public @torch_jit$async_dispatch_13_matmul_like_192x28x28x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_13_matmul_like_192x28x28x32_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x28x28xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x32xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<192x30x30xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 28, 28], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x28x28xf32>> -> tensor<32x28x28xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [192, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x32xf32>> -> tensor<192x32xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [192], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>> -> tensor<192xf32>
        %3 = tensor.empty() : tensor<192x28x28xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<192x28x28xf32>) -> tensor<192x28x28xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<32x28x28xf32>, tensor<192x32xf32>) outs(%4 : tensor<192x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<192x28x28xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<192x28x28xf32>, tensor<192xf32>) outs(%3 : tensor<192x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<192x28x28xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [192, 28, 28], strides = [1, 1, 1] : tensor<192x28x28xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<192x30x30xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_14 {
    flow.executable.export public @torch_jit$async_dispatch_14_conv_28x28x192x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_14_conv_28x28x192x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x30x30xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<192x784xf32, #encoding23>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [192, 30, 30], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x30x30xf32>> -> tensor<192x30x30xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [192, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x3x3xf32>> -> tensor<192x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [192], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>> -> tensor<192xf32>
        %3 = tensor.empty() : tensor<192x28x28xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<192x28x28xf32>) -> tensor<192x28x28xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<192x30x30xf32>, tensor<192x3x3xf32>) outs(%4 : tensor<192x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<192x28x28xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<192x28x28xf32>, tensor<192xf32>) outs(%3 : tensor<192x28x28xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<192x28x28xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<192x28x28xf32> into tensor<192x784xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<192x784xf32> -> tensor<192x784xf32, #encoding23>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [192, 784], strides = [1, 1] : tensor<192x784xf32, #encoding23> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<192x784xf32, #encoding23>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_15 {
    flow.executable.export public @torch_jit$async_dispatch_15_matmul_like_32x784x192_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_15_matmul_like_32x784x192_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x784xf32, #encoding23>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x192xf32, #encoding4>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x784xf32, #encoding24>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding46>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x784xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [192, 784], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x784xf32, #encoding23>> -> tensor<192x784xf32, #encoding23>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [32, 192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x192xf32, #encoding4>> -> tensor<32x192xf32, #encoding4>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [32, 784], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x784xf32, #encoding24>> -> tensor<32x784xf32, #encoding24>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [32], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32xf32, #encoding46>> -> tensor<32xf32, #encoding46>
        %4 = tensor.empty() : tensor<32x784xf32, #encoding24>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x784xf32, #encoding24>) -> tensor<32x784xf32, #encoding24>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<192x784xf32, #encoding23>, tensor<32x192xf32, #encoding4>) outs(%5 : tensor<32x784xf32, #encoding24>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<32x784xf32, #encoding24>
        %7 = linalg.generic {indexing_maps = [#map5, #map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%2, %6, %3 : tensor<32x784xf32, #encoding24>, tensor<32x784xf32, #encoding24>, tensor<32xf32, #encoding46>) outs(%4 : tensor<32x784xf32, #encoding24>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<32x784xf32, #encoding24>
        %8 = iree_encoding.unset_encoding %7 : tensor<32x784xf32, #encoding24> -> tensor<32x784xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %arg4, offsets = [0, 0], sizes = [32, 784], strides = [1, 1] : tensor<32x784xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x784xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_20 {
    flow.executable.export public @torch_jit$async_dispatch_20_conv_14x14x192x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_20_conv_14x14x192x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x30x30xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<192x196xf32, #encoding25>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [192, 30, 30], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x30x30xf32>> -> tensor<192x30x30xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [192, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x3x3xf32>> -> tensor<192x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [192], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192xf32>> -> tensor<192xf32>
        %3 = tensor.empty() : tensor<192x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<192x14x14xf32>) -> tensor<192x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map17, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<192x30x30xf32>, tensor<192x3x3xf32>) outs(%4 : tensor<192x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<192x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<192x14x14xf32>, tensor<192xf32>) outs(%3 : tensor<192x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<192x14x14xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<192x14x14xf32> into tensor<192x196xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<192x196xf32> -> tensor<192x196xf32, #encoding25>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [192, 196], strides = [1, 1] : tensor<192x196xf32, #encoding25> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<192x196xf32, #encoding25>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_21 {
    flow.executable.export public @torch_jit$async_dispatch_21_matmul_like_64x196x192_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_21_matmul_like_64x196x192_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x196xf32, #encoding25>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x192xf32, #encoding5>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32, #encoding47>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x196xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [192, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<192x196xf32, #encoding25>> -> tensor<192x196xf32, #encoding25>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x192xf32, #encoding5>> -> tensor<64x192xf32, #encoding5>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32, #encoding47>> -> tensor<64xf32, #encoding47>
        %3 = tensor.empty() : tensor<64x196xf32, #encoding26>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x196xf32, #encoding26>) -> tensor<64x196xf32, #encoding26>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<192x196xf32, #encoding25>, tensor<64x192xf32, #encoding5>) outs(%4 : tensor<64x196xf32, #encoding26>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<64x196xf32, #encoding26>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<64x196xf32, #encoding26>, tensor<64xf32, #encoding47>) outs(%3 : tensor<64x196xf32, #encoding26>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<64x196xf32, #encoding26>
        %7 = iree_encoding.unset_encoding %6 : tensor<64x196xf32, #encoding26> -> tensor<64x196xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [64, 196], strides = [1, 1] : tensor<64x196xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x196xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_22 {
    flow.executable.export public @torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x14x14xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x64xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<384x16x16xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 14, 14], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x14x14xf32>> -> tensor<64x14x14xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [384, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x64xf32>> -> tensor<384x64xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %3 = tensor.empty() : tensor<384x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<384x14x14xf32>) -> tensor<384x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<64x14x14xf32>, tensor<384x64xf32>) outs(%4 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<384x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<384x14x14xf32>, tensor<384xf32>) outs(%3 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<384x14x14xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [384, 14, 14], strides = [1, 1, 1] : tensor<384x14x14xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<384x16x16xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_23 {
    flow.executable.export public @torch_jit$async_dispatch_23_conv_14x14x384x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_23_conv_14x14x384x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x196xf32, #encoding27>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [384, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x16x16xf32>> -> tensor<384x16x16xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [384, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x3x3xf32>> -> tensor<384x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %3 = tensor.empty() : tensor<384x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<384x14x14xf32>) -> tensor<384x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<384x16x16xf32>, tensor<384x3x3xf32>) outs(%4 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<384x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<384x14x14xf32>, tensor<384xf32>) outs(%3 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<384x14x14xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<384x14x14xf32> into tensor<384x196xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<384x196xf32> -> tensor<384x196xf32, #encoding27>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [384, 196], strides = [1, 1] : tensor<384x196xf32, #encoding27> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x196xf32, #encoding27>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_24 {
    flow.executable.export public @torch_jit$async_dispatch_24_matmul_like_64x196x384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_24_matmul_like_64x196x384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x196xf32, #encoding27>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x384xf32, #encoding6>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x196xf32, #encoding28>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32, #encoding48>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x196xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [384, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x196xf32, #encoding27>> -> tensor<384x196xf32, #encoding27>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x384xf32, #encoding6>> -> tensor<64x384xf32, #encoding6>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [64, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x196xf32, #encoding28>> -> tensor<64x196xf32, #encoding28>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32, #encoding48>> -> tensor<64xf32, #encoding48>
        %4 = tensor.empty() : tensor<64x196xf32, #encoding28>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x196xf32, #encoding28>) -> tensor<64x196xf32, #encoding28>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<384x196xf32, #encoding27>, tensor<64x384xf32, #encoding6>) outs(%5 : tensor<64x196xf32, #encoding28>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<64x196xf32, #encoding28>
        %7 = linalg.generic {indexing_maps = [#map5, #map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%2, %6, %3 : tensor<64x196xf32, #encoding28>, tensor<64x196xf32, #encoding28>, tensor<64xf32, #encoding48>) outs(%4 : tensor<64x196xf32, #encoding28>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<64x196xf32, #encoding28>
        %8 = iree_encoding.unset_encoding %7 : tensor<64x196xf32, #encoding28> -> tensor<64x196xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %arg4, offsets = [0, 0], sizes = [64, 196], strides = [1, 1] : tensor<64x196xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x196xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_32 {
    flow.executable.export public @torch_jit$async_dispatch_32_conv_14x14x384x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_32_conv_14x14x384x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x196xf32, #encoding29>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [384, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x16x16xf32>> -> tensor<384x16x16xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [384, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x3x3xf32>> -> tensor<384x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
        %3 = tensor.empty() : tensor<384x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<384x14x14xf32>) -> tensor<384x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<384x16x16xf32>, tensor<384x3x3xf32>) outs(%4 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<384x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<384x14x14xf32>, tensor<384xf32>) outs(%3 : tensor<384x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<384x14x14xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<384x14x14xf32> into tensor<384x196xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<384x196xf32> -> tensor<384x196xf32, #encoding29>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [384, 196], strides = [1, 1] : tensor<384x196xf32, #encoding29> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x196xf32, #encoding29>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_33 {
    flow.executable.export public @torch_jit$async_dispatch_33_matmul_like_96x196x384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_33_matmul_like_96x196x384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x196xf32, #encoding29>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x384xf32, #encoding7>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32, #encoding49>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x196xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [384, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x196xf32, #encoding29>> -> tensor<384x196xf32, #encoding29>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [96, 384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x384xf32, #encoding7>> -> tensor<96x384xf32, #encoding7>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [96], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32, #encoding49>> -> tensor<96xf32, #encoding49>
        %3 = tensor.empty() : tensor<96x196xf32, #encoding30>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<96x196xf32, #encoding30>) -> tensor<96x196xf32, #encoding30>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<384x196xf32, #encoding29>, tensor<96x384xf32, #encoding7>) outs(%4 : tensor<96x196xf32, #encoding30>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<96x196xf32, #encoding30>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<96x196xf32, #encoding30>, tensor<96xf32, #encoding49>) outs(%3 : tensor<96x196xf32, #encoding30>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<96x196xf32, #encoding30>
        %7 = iree_encoding.unset_encoding %6 : tensor<96x196xf32, #encoding30> -> tensor<96x196xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [96, 196], strides = [1, 1] : tensor<96x196xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x196xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_34 {
    flow.executable.export public @torch_jit$async_dispatch_34_matmul_like_576x14x14x96_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_34_matmul_like_576x14x14x96_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x14x14xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x96xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<576x16x16xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [96, 14, 14], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x14x14xf32>> -> tensor<96x14x14xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [576, 96], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x96xf32>> -> tensor<576x96xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [576], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>> -> tensor<576xf32>
        %3 = tensor.empty() : tensor<576x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<576x14x14xf32>) -> tensor<576x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<96x14x14xf32>, tensor<576x96xf32>) outs(%4 : tensor<576x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<576x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<576x14x14xf32>, tensor<576xf32>) outs(%3 : tensor<576x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<576x14x14xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [576, 14, 14], strides = [1, 1, 1] : tensor<576x14x14xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<576x16x16xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_35 {
    flow.executable.export public @torch_jit$async_dispatch_35_conv_14x14x576x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_35_conv_14x14x576x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<576x196xf32, #encoding31>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [576, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x16x16xf32>> -> tensor<576x16x16xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [576, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3xf32>> -> tensor<576x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [576], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>> -> tensor<576xf32>
        %3 = tensor.empty() : tensor<576x14x14xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<576x14x14xf32>) -> tensor<576x14x14xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<576x16x16xf32>, tensor<576x3x3xf32>) outs(%4 : tensor<576x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<576x14x14xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<576x14x14xf32>, tensor<576xf32>) outs(%3 : tensor<576x14x14xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<576x14x14xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<576x14x14xf32> into tensor<576x196xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<576x196xf32> -> tensor<576x196xf32, #encoding31>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [576, 196], strides = [1, 1] : tensor<576x196xf32, #encoding31> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<576x196xf32, #encoding31>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_36 {
    flow.executable.export public @torch_jit$async_dispatch_36_matmul_like_96x196x576_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_36_matmul_like_96x196x576_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x196xf32, #encoding31>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x576xf32, #encoding8>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x196xf32, #encoding32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32, #encoding50>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x196xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [576, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x196xf32, #encoding31>> -> tensor<576x196xf32, #encoding31>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [96, 576], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x576xf32, #encoding8>> -> tensor<96x576xf32, #encoding8>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [96, 196], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96x196xf32, #encoding32>> -> tensor<96x196xf32, #encoding32>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [96], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<96xf32, #encoding50>> -> tensor<96xf32, #encoding50>
        %4 = tensor.empty() : tensor<96x196xf32, #encoding32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<96x196xf32, #encoding32>) -> tensor<96x196xf32, #encoding32>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<576x196xf32, #encoding31>, tensor<96x576xf32, #encoding8>) outs(%5 : tensor<96x196xf32, #encoding32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<96x196xf32, #encoding32>
        %7 = linalg.generic {indexing_maps = [#map5, #map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%2, %6, %3 : tensor<96x196xf32, #encoding32>, tensor<96x196xf32, #encoding32>, tensor<96xf32, #encoding50>) outs(%4 : tensor<96x196xf32, #encoding32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<96x196xf32, #encoding32>
        %8 = iree_encoding.unset_encoding %7 : tensor<96x196xf32, #encoding32> -> tensor<96x196xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %arg4, offsets = [0, 0], sizes = [96, 196], strides = [1, 1] : tensor<96x196xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<96x196xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_41 {
    flow.executable.export public @torch_jit$async_dispatch_41_conv_7x7x576x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_41_conv_7x7x576x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<576x49xf32, #encoding33>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [576, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x16x16xf32>> -> tensor<576x16x16xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [576, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x3x3xf32>> -> tensor<576x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [576], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576xf32>> -> tensor<576xf32>
        %3 = tensor.empty() : tensor<576x7x7xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<576x7x7xf32>) -> tensor<576x7x7xf32>
        %5 = linalg.generic {indexing_maps = [#map17, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<576x16x16xf32>, tensor<576x3x3xf32>) outs(%4 : tensor<576x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<576x7x7xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<576x7x7xf32>, tensor<576xf32>) outs(%3 : tensor<576x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<576x7x7xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<576x7x7xf32> into tensor<576x49xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<576x49xf32> -> tensor<576x49xf32, #encoding33>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [576, 49], strides = [1, 1] : tensor<576x49xf32, #encoding33> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<576x49xf32, #encoding33>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_42 {
    flow.executable.export public @torch_jit$async_dispatch_42_matmul_like_160x49x576_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_42_matmul_like_160x49x576_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x49xf32, #encoding33>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x576xf32, #encoding9>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160xf32, #encoding51>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<160x49xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [576, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<576x49xf32, #encoding33>> -> tensor<576x49xf32, #encoding33>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [160, 576], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x576xf32, #encoding9>> -> tensor<160x576xf32, #encoding9>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [160], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160xf32, #encoding51>> -> tensor<160xf32, #encoding51>
        %3 = tensor.empty() : tensor<160x49xf32, #encoding34>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<160x49xf32, #encoding34>) -> tensor<160x49xf32, #encoding34>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<576x49xf32, #encoding33>, tensor<160x576xf32, #encoding9>) outs(%4 : tensor<160x49xf32, #encoding34>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<160x49xf32, #encoding34>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<160x49xf32, #encoding34>, tensor<160xf32, #encoding51>) outs(%3 : tensor<160x49xf32, #encoding34>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<160x49xf32, #encoding34>
        %7 = iree_encoding.unset_encoding %6 : tensor<160x49xf32, #encoding34> -> tensor<160x49xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [160, 49], strides = [1, 1] : tensor<160x49xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<160x49xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_43 {
    flow.executable.export public @torch_jit$async_dispatch_43_matmul_like_960x7x7x160_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_43_matmul_like_960x7x7x160_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x7x7xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x160xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<960x9x9xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [160, 7, 7], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x7x7xf32>> -> tensor<160x7x7xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [960, 160], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x160xf32>> -> tensor<960x160xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [960], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>> -> tensor<960xf32>
        %3 = tensor.empty() : tensor<960x7x7xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<960x7x7xf32>) -> tensor<960x7x7xf32>
        %5 = linalg.generic {indexing_maps = [#map14, #map15, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<160x7x7xf32>, tensor<960x160xf32>) outs(%4 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.mulf %in, %in_1 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<960x7x7xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<960x7x7xf32>, tensor<960xf32>) outs(%3 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %7 = arith.addf %in, %in_1 : f32
          %8 = arith.cmpf ult, %7, %cst : f32
          %9 = arith.select %8, %cst, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst_0 : f32
          %11 = arith.select %10, %cst_0, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<960x7x7xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 1, 1], sizes = [960, 7, 7], strides = [1, 1, 1] : tensor<960x7x7xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<960x9x9xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_44 {
    flow.executable.export public @torch_jit$async_dispatch_44_conv_7x7x960x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_44_conv_7x7x960x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x9x9xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960x49xf32, #encoding35>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [960, 9, 9], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x9x9xf32>> -> tensor<960x9x9xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [960, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x3x3xf32>> -> tensor<960x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [960], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>> -> tensor<960xf32>
        %3 = tensor.empty() : tensor<960x7x7xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<960x7x7xf32>) -> tensor<960x7x7xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<960x9x9xf32>, tensor<960x3x3xf32>) outs(%4 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<960x7x7xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<960x7x7xf32>, tensor<960xf32>) outs(%3 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<960x7x7xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<960x7x7xf32> into tensor<960x49xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<960x49xf32> -> tensor<960x49xf32, #encoding35>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [960, 49], strides = [1, 1] : tensor<960x49xf32, #encoding35> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960x49xf32, #encoding35>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_45 {
    flow.executable.export public @torch_jit$async_dispatch_45_matmul_like_160x49x960_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_45_matmul_like_160x49x960_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x49xf32, #encoding35>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x960xf32, #encoding10>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x49xf32, #encoding36>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<160xf32, #encoding52>>, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<160x49xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [960, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x49xf32, #encoding35>> -> tensor<960x49xf32, #encoding35>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [160, 960], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x960xf32, #encoding10>> -> tensor<160x960xf32, #encoding10>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [160, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160x49xf32, #encoding36>> -> tensor<160x49xf32, #encoding36>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [160], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<160xf32, #encoding52>> -> tensor<160xf32, #encoding52>
        %4 = tensor.empty() : tensor<160x49xf32, #encoding36>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<160x49xf32, #encoding36>) -> tensor<160x49xf32, #encoding36>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<960x49xf32, #encoding35>, tensor<160x960xf32, #encoding10>) outs(%5 : tensor<160x49xf32, #encoding36>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<160x49xf32, #encoding36>
        %7 = linalg.generic {indexing_maps = [#map5, #map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%2, %6, %3 : tensor<160x49xf32, #encoding36>, tensor<160x49xf32, #encoding36>, tensor<160xf32, #encoding52>) outs(%4 : tensor<160x49xf32, #encoding36>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<160x49xf32, #encoding36>
        %8 = iree_encoding.unset_encoding %7 : tensor<160x49xf32, #encoding36> -> tensor<160x49xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %arg4, offsets = [0, 0], sizes = [160, 49], strides = [1, 1] : tensor<160x49xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<160x49xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_50 {
    flow.executable.export public @torch_jit$async_dispatch_50_conv_7x7x960x3x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_50_conv_7x7x960x3x3_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x9x9xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x3x3xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960x49xf32, #encoding37>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [960, 9, 9], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x9x9xf32>> -> tensor<960x9x9xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [960, 3, 3], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x3x3xf32>> -> tensor<960x3x3xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [960], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf32>> -> tensor<960xf32>
        %3 = tensor.empty() : tensor<960x7x7xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<960x7x7xf32>) -> tensor<960x7x7xf32>
        %5 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<960x9x9xf32>, tensor<960x3x3xf32>) outs(%4 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.mulf %in, %in_1 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<960x7x7xf32>
        %6 = linalg.generic {indexing_maps = [#map9, #map10, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %2 : tensor<960x7x7xf32>, tensor<960xf32>) outs(%3 : tensor<960x7x7xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.addf %in, %in_1 : f32
          %9 = arith.cmpf ult, %8, %cst : f32
          %10 = arith.select %9, %cst, %8 : f32
          %11 = arith.cmpf ugt, %10, %cst_0 : f32
          %12 = arith.select %11, %cst_0, %10 : f32
          linalg.yield %12 : f32
        } -> tensor<960x7x7xf32>
        %collapsed = tensor.collapse_shape %6 [[0], [1, 2]] : tensor<960x7x7xf32> into tensor<960x49xf32>
        %7 = iree_encoding.set_encoding %collapsed : tensor<960x49xf32> -> tensor<960x49xf32, #encoding37>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [960, 49], strides = [1, 1] : tensor<960x49xf32, #encoding37> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960x49xf32, #encoding37>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_51 {
    flow.executable.export public @torch_jit$async_dispatch_51_matmul_like_320x49x960_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_51_matmul_like_320x49x960_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x49xf32, #encoding37>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x960xf32, #encoding11>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<320xf32, #encoding53>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<320x49xf32, #encoding38>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [960, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960x49xf32, #encoding37>> -> tensor<960x49xf32, #encoding37>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [320, 960], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x960xf32, #encoding11>> -> tensor<320x960xf32, #encoding11>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [320], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320xf32, #encoding53>> -> tensor<320xf32, #encoding53>
        %3 = tensor.empty() : tensor<320x49xf32, #encoding39>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<320x49xf32, #encoding39>) -> tensor<320x49xf32, #encoding39>
        %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<960x49xf32, #encoding37>, tensor<320x960xf32, #encoding11>) outs(%4 : tensor<320x49xf32, #encoding39>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<320x49xf32, #encoding39>
        %6 = linalg.generic {indexing_maps = [#map5, #map3, #map5], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<320x49xf32, #encoding39>, tensor<320xf32, #encoding53>) outs(%3 : tensor<320x49xf32, #encoding39>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.addf %in, %in_0 : f32
          linalg.yield %9 : f32
        } -> tensor<320x49xf32, #encoding39>
        %7 = iree_encoding.unset_encoding %6 : tensor<320x49xf32, #encoding39> -> tensor<320x49xf32>
        %8 = iree_encoding.set_encoding %7 : tensor<320x49xf32> -> tensor<320x49xf32, #encoding38>
        iree_tensor_ext.dispatch.tensor.store %8, %arg3, offsets = [0, 0], sizes = [320, 49], strides = [1, 1] : tensor<320x49xf32, #encoding38> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<320x49xf32, #encoding38>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_52 {
    flow.executable.export public @torch_jit$async_dispatch_52_matmul_like_1280x49x320_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_52_matmul_like_1280x49x320_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x49xf32, #encoding38>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x320xf32, #encoding12>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1280x49xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [320, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320x49xf32, #encoding38>> -> tensor<320x49xf32, #encoding38>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1280, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x320xf32, #encoding12>> -> tensor<1280x320xf32, #encoding12>
        %2 = tensor.empty() : tensor<1280x49xf32, #encoding40>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1280x49xf32, #encoding40>) -> tensor<1280x49xf32, #encoding40>
        %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<320x49xf32, #encoding38>, tensor<1280x320xf32, #encoding12>) outs(%3 : tensor<1280x49xf32, #encoding40>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %6 = arith.mulf %in, %in_0 : f32
          %7 = arith.addf %out, %6 : f32
          linalg.yield %7 : f32
        } -> tensor<1280x49xf32, #encoding40>
        %5 = iree_encoding.unset_encoding %4 : tensor<1280x49xf32, #encoding40> -> tensor<1280x49xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 0], sizes = [1280, 49], strides = [1, 1] : tensor<1280x49xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1280x49xf32>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_53 {
    flow.executable.export public @torch_jit$async_dispatch_53_reduction_1280x49_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_53_reduction_1280x49_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x49xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1280xf32, #encoding41>>) {
        %cst = arith.constant 6.000000e+00 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %cst_1 = arith.constant 4.900000e+01 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1280, 49], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280x49xf32>> -> tensor<1280x49xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [1280], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %2 = tensor.empty() : tensor<1280xf32>
        %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<1280xf32>) -> tensor<1280xf32>
        %4 = linalg.generic {indexing_maps = [#map5, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%0, %1 : tensor<1280x49xf32>, tensor<1280xf32>) outs(%3 : tensor<1280xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %7 = arith.addf %in, %in_2 : f32
          %8 = arith.cmpf ult, %7, %cst_0 : f32
          %9 = arith.select %8, %cst_0, %7 : f32
          %10 = arith.cmpf ugt, %9, %cst : f32
          %11 = arith.select %10, %cst, %9 : f32
          %12 = arith.addf %out, %11 : f32
          linalg.yield %12 : f32
        } -> tensor<1280xf32>
        %5 = linalg.generic {indexing_maps = [#map18, #map18], iterator_types = ["parallel"]} ins(%4 : tensor<1280xf32>) outs(%2 : tensor<1280xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.divf %in, %cst_1 : f32
          linalg.yield %7 : f32
        } -> tensor<1280xf32>
        %6 = iree_encoding.set_encoding %5 : tensor<1280xf32> -> tensor<1280xf32, #encoding41>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0], sizes = [1280], strides = [1] : tensor<1280xf32, #encoding41> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1280xf32, #encoding41>>
        return
      }
    }
  }
  flow.executable private @torch_jit$async_dispatch_54 {
    flow.executable.export public @torch_jit$async_dispatch_54_matvec_like_1000x1280_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @torch_jit$async_dispatch_54_matvec_like_1000x1280_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280xf32, #encoding41>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x1280xf32, #encoding13>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000xf32, #encoding14>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [1280], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1280xf32, #encoding41>> -> tensor<1280xf32, #encoding41>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x1280xf32, #encoding13>> -> tensor<1000x1280xf32, #encoding13>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [1000], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000xf32, #encoding14>> -> tensor<1000xf32, #encoding14>
        %3 = tensor.empty() : tensor<1000xf32, #encoding14>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1000xf32, #encoding14>) -> tensor<1000xf32, #encoding14>
        %5 = linalg.generic {indexing_maps = [#map4, #map5, #map3], iterator_types = ["parallel", "reduction"]} ins(%0, %1 : tensor<1280xf32, #encoding41>, tensor<1000x1280xf32, #encoding13>) outs(%4 : tensor<1000xf32, #encoding14>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.mulf %in, %in_0 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<1000xf32, #encoding14>
        %6 = linalg.generic {indexing_maps = [#map18, #map18, #map18], iterator_types = ["parallel"]} ins(%5, %2 : tensor<1000xf32, #encoding14>, tensor<1000xf32, #encoding14>) outs(%3 : tensor<1000xf32, #encoding14>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<1000xf32, #encoding14>
        %7 = iree_encoding.unset_encoding %6 : tensor<1000xf32, #encoding14> -> tensor<1000xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0], sizes = [1000], strides = [1] : tensor<1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000xf32>>
        return
      }
    }
  }
  ...
  util.global private @__constant_tensor_384x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense80BC
  util.global private @__constant_tensor_384x3x3xf32_29 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = de2C8E
  util.global private @__constant_tensor_384x3x3xf32_30 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = deC1B47
  util.global private @__constant_tensor_384x3x3xf32_31 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = deEE431
  util.global private @__constant_tensor_192x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = denseB9BED31
  util.global private @__constant_tensor_192x3x3xf32_32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = deDE93EBF
  util.global private @__constant_tensor_192x3x3xf32_33 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = de89283BF
  util.global private @__constant_tensor_144x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense4C3F171
  util.global private @__constant_tensor_144x3x3xf32_34 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = deB40E2BE
  util.global private @__constant_tensor_96x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_32x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_32x3x3x3xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dens
  util.global private @__constant_tensor_96xf32_35 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<[
  util.global private @__constant_tensor_96xf32_36 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<[
  util.global private @__constant_tensor_144xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0x
  util.global private @__constant_tensor_144xf32_37 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_144xf32_38 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_144xf32_39 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_192xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0x
  util.global private @__constant_tensor_192xf32_40 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_192xf32_41 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_192xf32_42 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_192xf32_43 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_192xf32_44 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0x
  util.global private @__constant_tensor_384xf32_45 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_46 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_47 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_48 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_49 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_50 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_384xf32_51 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_576xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0x
  util.global private @__constant_tensor_576xf32_52 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_576xf32_53 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_576xf32_54 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_576xf32_55 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_576xf32_56 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_960xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0x
  util.global private @__constant_tensor_960xf32_57 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_960xf32_58 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_960xf32_59 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_960xf32_60 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_960xf32_61 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<
  util.global private @__constant_tensor_1280xf32 {inlining_policy = #util.inline.never, stream.affinity.default = #hal.device.affinity<@device_a>} = dense<"0EEC23B3BDE9803EBE2A36733C0BF94ABED5E04DBE5A09F8BDD96C2DBE2B9237BE34061EBED12145BE33E920BE25E50CBEDFC1ADBC7B791FBE370226BE1663693D1E3E7CBEFE922DBE4F304DBED06D2CBEBD0939BE1AF130BE7C90DABDFF96FFBD756305BE6AE910BEC38E18BE646607BEBC616ABE0AEC26BED5AD19BE66410FBE967804BE226EC7BD57E65CBEC13DC4BD474311BE90BD67BE23E376BE3D4117BE5C0EACBDC4C834BE2EAE1FBE425574BE952624BE"> : tensor<1280xf32>
  util.func public @torch_jit$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %__constant_tensor_960x160xf32 = util.global.load immutable @__constant_tensor_960x160xf32 : tensor<960x160xf32>
    %__constant_tensor_960x160xf32_15 = util.global.load immutable @__constant_tensor_960x160xf32_15 : tensor<960x160xf32>
    %__constant_tensor_960x160xf32_16 = util.global.load immutable @__constant_tensor_960x160xf32_16 : tensor<960x160xf32>
    %__constant_tensor_576x96xf32 = util.global.load immutable @__constant_tensor_576x96xf32 : tensor<576x96xf32>
    %__constant_tensor_576x96xf32_17 = util.global.load immutable @__constant_tensor_576x96xf32_17 : tensor<576x96xf32>
    %__constant_tensor_576x96xf32_18 = util.global.load immutable @__constant_tensor_576x96xf32_18 : tensor<576x96xf32>
    %__constant_tensor_384x64xf32 = util.global.load immutable @__constant_tensor_384x64xf32 : tensor<384x64xf32>
    %__constant_tensor_384x64xf32_19 = util.global.load immutable @__constant_tensor_384x64xf32_19 : tensor<384x64xf32>
    %__constant_tensor_384x64xf32_20 = util.global.load immutable @__constant_tensor_384x64xf32_20 : tensor<384x64xf32>
    %__constant_tensor_384x64xf32_21 = util.global.load immutable @__constant_tensor_384x64xf32_21 : tensor<384x64xf32>
    %__constant_tensor_192x32xf32 = util.global.load immutable @__constant_tensor_192x32xf32 : tensor<192x32xf32>
    %__constant_tensor_192x32xf32_22 = util.global.load immutable @__constant_tensor_192x32xf32_22 : tensor<192x32xf32>
    %__constant_tensor_192x32xf32_23 = util.global.load immutable @__constant_tensor_192x32xf32_23 : tensor<192x32xf32>
    %__constant_tensor_144x24xf32 = util.global.load immutable @__constant_tensor_144x24xf32 : tensor<144x24xf32>
    %__constant_tensor_144x24xf32_24 = util.global.load immutable @__constant_tensor_144x24xf32_24 : tensor<144x24xf32>
    %__constant_tensor_96x16xf32 = util.global.load immutable @__constant_tensor_96x16xf32 : tensor<96x16xf32>
    %__constant_tensor_960x3x3xf32 = util.global.load immutable @__constant_tensor_960x3x3xf32 : tensor<960x3x3xf32>
    %__constant_tensor_960x3x3xf32_25 = util.global.load immutable @__constant_tensor_960x3x3xf32_25 : tensor<960x3x3xf32>
    %__constant_tensor_960x3x3xf32_26 = util.global.load immutable @__constant_tensor_960x3x3xf32_26 : tensor<960x3x3xf32>
    %__constant_tensor_576x3x3xf32 = util.global.load immutable @__constant_tensor_576x3x3xf32 : tensor<576x3x3xf32>
    %__constant_tensor_576x3x3xf32_27 = util.global.load immutable @__constant_tensor_576x3x3xf32_27 : tensor<576x3x3xf32>
    %__constant_tensor_576x3x3xf32_28 = util.global.load immutable @__constant_tensor_576x3x3xf32_28 : tensor<576x3x3xf32>
    %__constant_tensor_384x3x3xf32 = util.global.load immutable @__constant_tensor_384x3x3xf32 : tensor<384x3x3xf32>
    %__constant_tensor_384x3x3xf32_29 = util.global.load immutable @__constant_tensor_384x3x3xf32_29 : tensor<384x3x3xf32>
    %__constant_tensor_384x3x3xf32_30 = util.global.load immutable @__constant_tensor_384x3x3xf32_30 : tensor<384x3x3xf32>
    %__constant_tensor_384x3x3xf32_31 = util.global.load immutable @__constant_tensor_384x3x3xf32_31 : tensor<384x3x3xf32>
    %__constant_tensor_192x3x3xf32 = util.global.load immutable @__constant_tensor_192x3x3xf32 : tensor<192x3x3xf32>
    %__constant_tensor_192x3x3xf32_32 = util.global.load immutable @__constant_tensor_192x3x3xf32_32 : tensor<192x3x3xf32>
    %__constant_tensor_192x3x3xf32_33 = util.global.load immutable @__constant_tensor_192x3x3xf32_33 : tensor<192x3x3xf32>
    %__constant_tensor_144x3x3xf32 = util.global.load immutable @__constant_tensor_144x3x3xf32 : tensor<144x3x3xf32>
    %__constant_tensor_144x3x3xf32_34 = util.global.load immutable @__constant_tensor_144x3x3xf32_34 : tensor<144x3x3xf32>
    %__constant_tensor_96x3x3xf32 = util.global.load immutable @__constant_tensor_96x3x3xf32 : tensor<96x3x3xf32>
    %__constant_tensor_32x3x3xf32 = util.global.load immutable @__constant_tensor_32x3x3xf32 : tensor<32x3x3xf32>
    %__constant_tensor_32x3x3x3xf32 = util.global.load immutable @__constant_tensor_32x3x3x3xf32 : tensor<32x3x3x3xf32>
    %__constant_tensor_96xf32_35 = util.global.load immutable @__constant_tensor_96xf32_35 : tensor<96xf32>
    %__constant_tensor_96xf32_36 = util.global.load immutable @__constant_tensor_96xf32_36 : tensor<96xf32>
    %__constant_tensor_144xf32 = util.global.load immutable @__constant_tensor_144xf32 : tensor<144xf32>
    %__constant_tensor_144xf32_37 = util.global.load immutable @__constant_tensor_144xf32_37 : tensor<144xf32>
    %__constant_tensor_144xf32_38 = util.global.load immutable @__constant_tensor_144xf32_38 : tensor<144xf32>
    %__constant_tensor_144xf32_39 = util.global.load immutable @__constant_tensor_144xf32_39 : tensor<144xf32>
    %__constant_tensor_192xf32 = util.global.load immutable @__constant_tensor_192xf32 : tensor<192xf32>
    %__constant_tensor_192xf32_40 = util.global.load immutable @__constant_tensor_192xf32_40 : tensor<192xf32>
    %__constant_tensor_192xf32_41 = util.global.load immutable @__constant_tensor_192xf32_41 : tensor<192xf32>
    %__constant_tensor_192xf32_42 = util.global.load immutable @__constant_tensor_192xf32_42 : tensor<192xf32>
    %__constant_tensor_192xf32_43 = util.global.load immutable @__constant_tensor_192xf32_43 : tensor<192xf32>
    %__constant_tensor_192xf32_44 = util.global.load immutable @__constant_tensor_192xf32_44 : tensor<192xf32>
    %__constant_tensor_384xf32 = util.global.load immutable @__constant_tensor_384xf32 : tensor<384xf32>
    %__constant_tensor_384xf32_45 = util.global.load immutable @__constant_tensor_384xf32_45 : tensor<384xf32>
    %__constant_tensor_384xf32_46 = util.global.load immutable @__constant_tensor_384xf32_46 : tensor<384xf32>
    %__constant_tensor_384xf32_47 = util.global.load immutable @__constant_tensor_384xf32_47 : tensor<384xf32>
    %__constant_tensor_384xf32_48 = util.global.load immutable @__constant_tensor_384xf32_48 : tensor<384xf32>
    %__constant_tensor_384xf32_49 = util.global.load immutable @__constant_tensor_384xf32_49 : tensor<384xf32>
    %__constant_tensor_384xf32_50 = util.global.load immutable @__constant_tensor_384xf32_50 : tensor<384xf32>
    %__constant_tensor_384xf32_51 = util.global.load immutable @__constant_tensor_384xf32_51 : tensor<384xf32>
    %__constant_tensor_576xf32 = util.global.load immutable @__constant_tensor_576xf32 : tensor<576xf32>
    %__constant_tensor_576xf32_52 = util.global.load immutable @__constant_tensor_576xf32_52 : tensor<576xf32>
    %__constant_tensor_576xf32_53 = util.global.load immutable @__constant_tensor_576xf32_53 : tensor<576xf32>
    %__constant_tensor_576xf32_54 = util.global.load immutable @__constant_tensor_576xf32_54 : tensor<576xf32>
    %__constant_tensor_576xf32_55 = util.global.load immutable @__constant_tensor_576xf32_55 : tensor<576xf32>
    %__constant_tensor_576xf32_56 = util.global.load immutable @__constant_tensor_576xf32_56 : tensor<576xf32>
    %__constant_tensor_960xf32 = util.global.load immutable @__constant_tensor_960xf32 : tensor<960xf32>
    %__constant_tensor_960xf32_57 = util.global.load immutable @__constant_tensor_960xf32_57 : tensor<960xf32>
    %__constant_tensor_960xf32_58 = util.global.load immutable @__constant_tensor_960xf32_58 : tensor<960xf32>
    %__constant_tensor_960xf32_59 = util.global.load immutable @__constant_tensor_960xf32_59 : tensor<960xf32>
    %__constant_tensor_960xf32_60 = util.global.load immutable @__constant_tensor_960xf32_60 : tensor<960xf32>
    %__constant_tensor_960xf32_61 = util.global.load immutable @__constant_tensor_960xf32_61 : tensor<960xf32>
    %__constant_tensor_1280xf32 = util.global.load immutable @__constant_tensor_1280xf32 : tensor<1280xf32>
    %__hoisted_tensor_16x32xf32__encoded = util.global.load immutable @__hoisted_tensor_16x32xf32__encoded : tensor<16x32xf32, #encoding>
    %__hoisted_tensor_16xf32__encoded = util.global.load immutable @__hoisted_tensor_16xf32__encoded : tensor<16xf32, #encoding42>
    %__hoisted_tensor_24x96xf32__encoded = util.global.load immutable @__hoisted_tensor_24x96xf32__encoded : tensor<24x96xf32, #encoding1>
    %__hoisted_tensor_24xf32__encoded = util.global.load immutable @__hoisted_tensor_24xf32__encoded : tensor<24xf32, #encoding43>
    %__hoisted_tensor_24x144xf32__encoded = util.global.load immutable @__hoisted_tensor_24x144xf32__encoded : tensor<24x144xf32, #encoding2>
    %__hoisted_tensor_24xf32__encoded_0 = util.global.load immutable @__hoisted_tensor_24xf32__encoded_0 : tensor<24xf32, #encoding44>
    %__hoisted_tensor_32x144xf32__encoded = util.global.load immutable @__hoisted_tensor_32x144xf32__encoded : tensor<32x144xf32, #encoding3>
    %__hoisted_tensor_32xf32__encoded = util.global.load immutable @__hoisted_tensor_32xf32__encoded : tensor<32xf32, #encoding45>
    %__hoisted_tensor_32x192xf32__encoded = util.global.load immutable @__hoisted_tensor_32x192xf32__encoded : tensor<32x192xf32, #encoding4>
    %__hoisted_tensor_32xf32__encoded_1 = util.global.load immutable @__hoisted_tensor_32xf32__encoded_1 : tensor<32xf32, #encoding46>
    %__hoisted_tensor_32x192xf32__encoded_2 = util.global.load immutable @__hoisted_tensor_32x192xf32__encoded_2 : tensor<32x192xf32, #encoding4>
    %__hoisted_tensor_32xf32__encoded_3 = util.global.load immutable @__hoisted_tensor_32xf32__encoded_3 : tensor<32xf32, #encoding46>
    %__hoisted_tensor_64x192xf32__encoded = util.global.load immutable @__hoisted_tensor_64x192xf32__encoded : tensor<64x192xf32, #encoding5>
    %__hoisted_tensor_64xf32__encoded = util.global.load immutable @__hoisted_tensor_64xf32__encoded : tensor<64xf32, #encoding47>
    %__hoisted_tensor_64x384xf32__encoded = util.global.load immutable @__hoisted_tensor_64x384xf32__encoded : tensor<64x384xf32, #encoding6>
    %__hoisted_tensor_64xf32__encoded_4 = util.global.load immutable @__hoisted_tensor_64xf32__encoded_4 : tensor<64xf32, #encoding48>
    %__hoisted_tensor_64x384xf32__encoded_5 = util.global.load immutable @__hoisted_tensor_64x384xf32__encoded_5 : tensor<64x384xf32, #encoding6>
    %__hoisted_tensor_64xf32__encoded_6 = util.global.load immutable @__hoisted_tensor_64xf32__encoded_6 : tensor<64xf32, #encoding48>
    %__hoisted_tensor_64x384xf32__encoded_7 = util.global.load immutable @__hoisted_tensor_64x384xf32__encoded_7 : tensor<64x384xf32, #encoding6>
    %__hoisted_tensor_64xf32__encoded_8 = util.global.load immutable @__hoisted_tensor_64xf32__encoded_8 : tensor<64xf32, #encoding48>
    %__hoisted_tensor_96x384xf32__encoded = util.global.load immutable @__hoisted_tensor_96x384xf32__encoded : tensor<96x384xf32, #encoding7>
    %__hoisted_tensor_96xf32__encoded = util.global.load immutable @__hoisted_tensor_96xf32__encoded : tensor<96xf32, #encoding49>
    %__hoisted_tensor_96x576xf32__encoded = util.global.load immutable @__hoisted_tensor_96x576xf32__encoded : tensor<96x576xf32, #encoding8>
    %__hoisted_tensor_96xf32__encoded_9 = util.global.load immutable @__hoisted_tensor_96xf32__encoded_9 : tensor<96xf32, #encoding50>
    %__hoisted_tensor_96x576xf32__encoded_10 = util.global.load immutable @__hoisted_tensor_96x576xf32__encoded_10 : tensor<96x576xf32, #encoding8>
    %__hoisted_tensor_96xf32__encoded_11 = util.global.load immutable @__hoisted_tensor_96xf32__encoded_11 : tensor<96xf32, #encoding50>
    %__hoisted_tensor_160x576xf32__encoded = util.global.load immutable @__hoisted_tensor_160x576xf32__encoded : tensor<160x576xf32, #encoding9>
    %__hoisted_tensor_160xf32__encoded = util.global.load immutable @__hoisted_tensor_160xf32__encoded : tensor<160xf32, #encoding51>
    %__hoisted_tensor_160x960xf32__encoded = util.global.load immutable @__hoisted_tensor_160x960xf32__encoded : tensor<160x960xf32, #encoding10>
    %__hoisted_tensor_160xf32__encoded_12 = util.global.load immutable @__hoisted_tensor_160xf32__encoded_12 : tensor<160xf32, #encoding52>
    %__hoisted_tensor_160x960xf32__encoded_13 = util.global.load immutable @__hoisted_tensor_160x960xf32__encoded_13 : tensor<160x960xf32, #encoding10>
    %__hoisted_tensor_160xf32__encoded_14 = util.global.load immutable @__hoisted_tensor_160xf32__encoded_14 : tensor<160xf32, #encoding52>
    %__hoisted_tensor_320x960xf32__encoded = util.global.load immutable @__hoisted_tensor_320x960xf32__encoded : tensor<320x960xf32, #encoding11>
    %__hoisted_tensor_320xf32__encoded = util.global.load immutable @__hoisted_tensor_320xf32__encoded : tensor<320xf32, #encoding53>
    %__hoisted_tensor_1280x320xf32__encoded = util.global.load immutable @__hoisted_tensor_1280x320xf32__encoded : tensor<1280x320xf32, #encoding12>
    %__hoisted_tensor_1000x1280xf32__encoded = util.global.load immutable @__hoisted_tensor_1000x1280xf32__encoded : tensor<1000x1280xf32, #encoding13>
    %__hoisted_tensor_1000xf32__encoded = util.global.load immutable @__hoisted_tensor_1000xf32__encoded : tensor<1000xf32, #encoding14>
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<1x3x224x224xf32>
    %1 = flow.tensor.reshape %0 : tensor<1x3x224x224xf32> -> tensor<3x224x224xf32>
    %2 = flow.tensor.splat %cst : tensor<3x226x226xf32>
%3 = flow.dispatch @torch_jit$async_dispatch_0::@torch_jit$async_dispatch_0_slow_memcpy(%1, %2) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<3x224x224xf32>, tensor<3x226x226xf32>) -> %2
    %4 = flow.tensor.splat %cst : tensor<32x114x114xf32>
%5 = flow.dispatch @torch_jit$async_dispatch_1::@torch_jit$async_dispatch_1_conv_32x112x112x3x3x3_f32(%3, %__constant_tensor_32x3x3x3xf32, %4) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<3x226x226xf32>, tensor<32x3x3x3xf32>, tensor<32x114x114xf32>) -> %4
%6 = flow.dispatch @torch_jit$async_dispatch_2::@torch_jit$async_dispatch_2_conv_112x112x32x3x3_f32(%5, %__constant_tensor_32x3x3xf32) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<32x114x114xf32>, tensor<32x3x3xf32>) -> tensor<32x112x112xf32>
    %7 = flow.tensor.reshape %6 : tensor<32x112x112xf32> -> tensor<32x12544xf32>
    %8 = flow.tensor.encode %7 : tensor<32x12544xf32> -> tensor<32x12544xf32, #encoding15>
%9 = flow.dispatch @torch_jit$async_dispatch_3::@torch_jit$async_dispatch_3_matmul_like_16x12544x32_f32(%8, %__hoisted_tensor_16x32xf32__encoded, %__hoisted_tensor_16xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<32x12544xf32, #encoding15>, tensor<16x32xf32, #encoding>, tensor<16xf32, #encoding42>) -> tensor<16x12544xf32>
    %10 = flow.tensor.reshape %9 : tensor<16x12544xf32> -> tensor<16x112x112xf32>
    %11 = flow.tensor.splat %cst : tensor<96x114x114xf32>
%12 = flow.dispatch @torch_jit$async_dispatch_4::@torch_jit$async_dispatch_4_matmul_like_96x112x112x16_f32(%10, %__constant_tensor_96x16xf32, %__constant_tensor_96xf32_35, %11) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16x112x112xf32>, tensor<96x16xf32>, tensor<96xf32>, tensor<96x114x114xf32>) -> %11
%13 = flow.dispatch @torch_jit$async_dispatch_5::@torch_jit$async_dispatch_5_conv_56x56x96x3x3_f32(%12, %__constant_tensor_96x3x3xf32, %__constant_tensor_96xf32_36) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<96x114x114xf32>, tensor<96x3x3xf32>, tensor<96xf32>) -> tensor<96x3136xf32, #encoding17>
%14 = flow.dispatch @torch_jit$async_dispatch_6::@torch_jit$async_dispatch_6_matmul_like_24x3136x96_f32(%13, %__hoisted_tensor_24x96xf32__encoded, %__hoisted_tensor_24xf32__encoded) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<96x3136xf32, #encoding17>, tensor<24x96xf32, #encoding1>, tensor<24xf32, #encoding43>) -> tensor<24x3136xf32>
    %15 = flow.tensor.reshape %14 : tensor<24x3136xf32> -> tensor<24x56x56xf32>
    %16 = flow.tensor.splat %cst : tensor<144x58x58xf32>
%17 = flow.dispatch @torch_jit$async_dispatch_7::@torch_jit$async_dispatch_7_matmul_like_144x56x56x24_f32(%15, %__constant_tensor_144x24xf32_24, %__constant_tensor_144xf32, %16) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<24x56x56xf32>, tensor<144x24xf32>, tensor<144xf32>, tensor<144x58x58xf32>) -> %16
%18 = flow.dispatch @torch_jit$async_dispatch_8::@torch_jit$async_dispatch_8_conv_56x56x144x3x3_f32(%17, %__constant_tensor_144x3x3xf32_34, %__constant_tensor_144xf32_37) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<144x58x58xf32>, tensor<144x3x3xf32>, tensor<144xf32>) -> tensor<144x3136xf32, #encoding19>
    %19 = flow.tensor.encode %14 : tensor<24x3136xf32> -> tensor<24x3136xf32, #encoding20>
%20 = flow.dispatch @torch_jit$async_dispatch_9::@torch_jit$async_dispatch_9_matmul_like_24x3136x144_f32(%18, %__hoisted_tensor_24x144xf32__encoded, %19, %__hoisted_tensor_24xf32__encoded_0) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<144x3136xf32, #encoding19>, tensor<24x144xf32, #encoding2>, tensor<24x3136xf32, #encoding20>, tensor<24xf32, #encoding44>) -> tensor<24x3136xf32>
    %21 = flow.tensor.reshape %20 : tensor<24x3136xf32> -> tensor<24x56x56xf32>
%22 = flow.dispatch @torch_jit$async_dispatch_7::@torch_jit$async_dispatch_7_matmul_like_144x56x56x24_f32(%21, %__constant_tensor_144x24xf32, %__constant_tensor_144xf32_38, %16) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<24x56x56xf32>, tensor<144x24xf32>, tensor<144xf32>, tensor<144x58x58xf32>) -> %16
%23 = flow.dispatch @torch_jit$async_dispatch_11::@torch_jit$async_dispatch_11_conv_28x28x144x3x3_f32(%22, %__constant_tensor_144x3x3xf32, %__constant_tensor_144xf32_39) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<144x58x58xf32>, tensor<144x3x3xf32>, tensor<144xf32>) -> tensor<144x784xf32, #encoding21>
%24 = flow.dispatch @torch_jit$async_dispatch_12::@torch_jit$async_dispatch_12_matmul_like_32x784x144_f32(%23, %__hoisted_tensor_32x144xf32__encoded, %__hoisted_tensor_32xf32__encoded) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<144x784xf32, #encoding21>, tensor<32x144xf32, #encoding3>, tensor<32xf32, #encoding45>) -> tensor<32x784xf32>
    %25 = flow.tensor.reshape %24 : tensor<32x784xf32> -> tensor<32x28x28xf32>
    %26 = flow.tensor.splat %cst : tensor<192x30x30xf32>
%27 = flow.dispatch @torch_jit$async_dispatch_13::@torch_jit$async_dispatch_13_matmul_like_192x28x28x32_f32(%25, %__constant_tensor_192x32xf32_23, %__constant_tensor_192xf32, %26) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<32x28x28xf32>, tensor<192x32xf32>, tensor<192xf32>, tensor<192x30x30xf32>) -> %26
%28 = flow.dispatch @torch_jit$async_dispatch_14::@torch_jit$async_dispatch_14_conv_28x28x192x3x3_f32(%27, %__constant_tensor_192x3x3xf32_33, %__constant_tensor_192xf32_40) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<192x30x30xf32>, tensor<192x3x3xf32>, tensor<192xf32>) -> tensor<192x784xf32, #encoding23>
    %29 = flow.tensor.encode %24 : tensor<32x784xf32> -> tensor<32x784xf32, #encoding24>
%30 = flow.dispatch @torch_jit$async_dispatch_15::@torch_jit$async_dispatch_15_matmul_like_32x784x192_f32(%28, %__hoisted_tensor_32x192xf32__encoded, %29, %__hoisted_tensor_32xf32__encoded_1) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<192x784xf32, #encoding23>, tensor<32x192xf32, #encoding4>, tensor<32x784xf32, #encoding24>, tensor<32xf32, #encoding46>) -> tensor<32x784xf32>
    %31 = flow.tensor.reshape %30 : tensor<32x784xf32> -> tensor<32x28x28xf32>
%32 = flow.dispatch @torch_jit$async_dispatch_13::@torch_jit$async_dispatch_13_matmul_like_192x28x28x32_f32(%31, %__constant_tensor_192x32xf32_22, %__constant_tensor_192xf32_41, %26) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<32x28x28xf32>, tensor<192x32xf32>, tensor<192xf32>, tensor<192x30x30xf32>) -> %26
%33 = flow.dispatch @torch_jit$async_dispatch_14::@torch_jit$async_dispatch_14_conv_28x28x192x3x3_f32(%32, %__constant_tensor_192x3x3xf32_32, %__constant_tensor_192xf32_42) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<192x30x30xf32>, tensor<192x3x3xf32>, tensor<192xf32>) -> tensor<192x784xf32, #encoding23>
    %34 = flow.tensor.encode %30 : tensor<32x784xf32> -> tensor<32x784xf32, #encoding24>
%35 = flow.dispatch @torch_jit$async_dispatch_15::@torch_jit$async_dispatch_15_matmul_like_32x784x192_f32(%33, %__hoisted_tensor_32x192xf32__encoded_2, %34, %__hoisted_tensor_32xf32__encoded_3) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<192x784xf32, #encoding23>, tensor<32x192xf32, #encoding4>, tensor<32x784xf32, #encoding24>, tensor<32xf32, #encoding46>) -> tensor<32x784xf32>
    %36 = flow.tensor.reshape %35 : tensor<32x784xf32> -> tensor<32x28x28xf32>
%37 = flow.dispatch @torch_jit$async_dispatch_13::@torch_jit$async_dispatch_13_matmul_like_192x28x28x32_f32(%36, %__constant_tensor_192x32xf32, %__constant_tensor_192xf32_43, %26) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<32x28x28xf32>, tensor<192x32xf32>, tensor<192xf32>, tensor<192x30x30xf32>) -> %26
%38 = flow.dispatch @torch_jit$async_dispatch_20::@torch_jit$async_dispatch_20_conv_14x14x192x3x3_f32(%37, %__constant_tensor_192x3x3xf32, %__constant_tensor_192xf32_44) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<192x30x30xf32>, tensor<192x3x3xf32>, tensor<192xf32>) -> tensor<192x196xf32, #encoding25>
%39 = flow.dispatch @torch_jit$async_dispatch_21::@torch_jit$async_dispatch_21_matmul_like_64x196x192_f32(%38, %__hoisted_tensor_64x192xf32__encoded, %__hoisted_tensor_64xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<192x196xf32, #encoding25>, tensor<64x192xf32, #encoding5>, tensor<64xf32, #encoding47>) -> tensor<64x196xf32>
    %40 = flow.tensor.reshape %39 : tensor<64x196xf32> -> tensor<64x14x14xf32>
    %41 = flow.tensor.splat %cst : tensor<384x16x16xf32>
%42 = flow.dispatch @torch_jit$async_dispatch_22::@torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32(%40, %__constant_tensor_384x64xf32_21, %__constant_tensor_384xf32, %41) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<64x14x14xf32>, tensor<384x64xf32>, tensor<384xf32>, tensor<384x16x16xf32>) -> %41
%43 = flow.dispatch @torch_jit$async_dispatch_23::@torch_jit$async_dispatch_23_conv_14x14x384x3x3_f32(%42, %__constant_tensor_384x3x3xf32_31, %__constant_tensor_384xf32_45) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x16x16xf32>, tensor<384x3x3xf32>, tensor<384xf32>) -> tensor<384x196xf32, #encoding27>
    %44 = flow.tensor.encode %39 : tensor<64x196xf32> -> tensor<64x196xf32, #encoding28>
%45 = flow.dispatch @torch_jit$async_dispatch_24::@torch_jit$async_dispatch_24_matmul_like_64x196x384_f32(%43, %__hoisted_tensor_64x384xf32__encoded, %44, %__hoisted_tensor_64xf32__encoded_4) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x196xf32, #encoding27>, tensor<64x384xf32, #encoding6>, tensor<64x196xf32, #encoding28>, tensor<64xf32, #encoding48>) -> tensor<64x196xf32>
    %46 = flow.tensor.reshape %45 : tensor<64x196xf32> -> tensor<64x14x14xf32>
%47 = flow.dispatch @torch_jit$async_dispatch_22::@torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32(%46, %__constant_tensor_384x64xf32_20, %__constant_tensor_384xf32_46, %41) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<64x14x14xf32>, tensor<384x64xf32>, tensor<384xf32>, tensor<384x16x16xf32>) -> %41
%48 = flow.dispatch @torch_jit$async_dispatch_23::@torch_jit$async_dispatch_23_conv_14x14x384x3x3_f32(%47, %__constant_tensor_384x3x3xf32_30, %__constant_tensor_384xf32_47) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x16x16xf32>, tensor<384x3x3xf32>, tensor<384xf32>) -> tensor<384x196xf32, #encoding27>
    %49 = flow.tensor.encode %45 : tensor<64x196xf32> -> tensor<64x196xf32, #encoding28>
%50 = flow.dispatch @torch_jit$async_dispatch_24::@torch_jit$async_dispatch_24_matmul_like_64x196x384_f32(%48, %__hoisted_tensor_64x384xf32__encoded_5, %49, %__hoisted_tensor_64xf32__encoded_6) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x196xf32, #encoding27>, tensor<64x384xf32, #encoding6>, tensor<64x196xf32, #encoding28>, tensor<64xf32, #encoding48>) -> tensor<64x196xf32>
    %51 = flow.tensor.reshape %50 : tensor<64x196xf32> -> tensor<64x14x14xf32>
%52 = flow.dispatch @torch_jit$async_dispatch_22::@torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32(%51, %__constant_tensor_384x64xf32_19, %__constant_tensor_384xf32_48, %41) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<64x14x14xf32>, tensor<384x64xf32>, tensor<384xf32>, tensor<384x16x16xf32>) -> %41
%53 = flow.dispatch @torch_jit$async_dispatch_23::@torch_jit$async_dispatch_23_conv_14x14x384x3x3_f32(%52, %__constant_tensor_384x3x3xf32_29, %__constant_tensor_384xf32_49) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x16x16xf32>, tensor<384x3x3xf32>, tensor<384xf32>) -> tensor<384x196xf32, #encoding27>
    %54 = flow.tensor.encode %50 : tensor<64x196xf32> -> tensor<64x196xf32, #encoding28>
%55 = flow.dispatch @torch_jit$async_dispatch_24::@torch_jit$async_dispatch_24_matmul_like_64x196x384_f32(%53, %__hoisted_tensor_64x384xf32__encoded_7, %54, %__hoisted_tensor_64xf32__encoded_8) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x196xf32, #encoding27>, tensor<64x384xf32, #encoding6>, tensor<64x196xf32, #encoding28>, tensor<64xf32, #encoding48>) -> tensor<64x196xf32>
    %56 = flow.tensor.reshape %55 : tensor<64x196xf32> -> tensor<64x14x14xf32>
%57 = flow.dispatch @torch_jit$async_dispatch_22::@torch_jit$async_dispatch_22_matmul_like_384x14x14x64_f32(%56, %__constant_tensor_384x64xf32, %__constant_tensor_384xf32_50, %41) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<64x14x14xf32>, tensor<384x64xf32>, tensor<384xf32>, tensor<384x16x16xf32>) -> %41
%58 = flow.dispatch @torch_jit$async_dispatch_32::@torch_jit$async_dispatch_32_conv_14x14x384x3x3_f32(%57, %__constant_tensor_384x3x3xf32, %__constant_tensor_384xf32_51) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x16x16xf32>, tensor<384x3x3xf32>, tensor<384xf32>) -> tensor<384x196xf32, #encoding29>
%59 = flow.dispatch @torch_jit$async_dispatch_33::@torch_jit$async_dispatch_33_matmul_like_96x196x384_f32(%58, %__hoisted_tensor_96x384xf32__encoded, %__hoisted_tensor_96xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<384x196xf32, #encoding29>, tensor<96x384xf32, #encoding7>, tensor<96xf32, #encoding49>) -> tensor<96x196xf32>
    %60 = flow.tensor.reshape %59 : tensor<96x196xf32> -> tensor<96x14x14xf32>
    %61 = flow.tensor.splat %cst : tensor<576x16x16xf32>
%62 = flow.dispatch @torch_jit$async_dispatch_34::@torch_jit$async_dispatch_34_matmul_like_576x14x14x96_f32(%60, %__constant_tensor_576x96xf32_18, %__constant_tensor_576xf32, %61) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<96x14x14xf32>, tensor<576x96xf32>, tensor<576xf32>, tensor<576x16x16xf32>) -> %61
%63 = flow.dispatch @torch_jit$async_dispatch_35::@torch_jit$async_dispatch_35_conv_14x14x576x3x3_f32(%62, %__constant_tensor_576x3x3xf32_28, %__constant_tensor_576xf32_52) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x16x16xf32>, tensor<576x3x3xf32>, tensor<576xf32>) -> tensor<576x196xf32, #encoding31>
    %64 = flow.tensor.encode %59 : tensor<96x196xf32> -> tensor<96x196xf32, #encoding32>
%65 = flow.dispatch @torch_jit$async_dispatch_36::@torch_jit$async_dispatch_36_matmul_like_96x196x576_f32(%63, %__hoisted_tensor_96x576xf32__encoded, %64, %__hoisted_tensor_96xf32__encoded_9) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x196xf32, #encoding31>, tensor<96x576xf32, #encoding8>, tensor<96x196xf32, #encoding32>, tensor<96xf32, #encoding50>) -> tensor<96x196xf32>
    %66 = flow.tensor.reshape %65 : tensor<96x196xf32> -> tensor<96x14x14xf32>
%67 = flow.dispatch @torch_jit$async_dispatch_34::@torch_jit$async_dispatch_34_matmul_like_576x14x14x96_f32(%66, %__constant_tensor_576x96xf32_17, %__constant_tensor_576xf32_53, %61) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<96x14x14xf32>, tensor<576x96xf32>, tensor<576xf32>, tensor<576x16x16xf32>) -> %61
%68 = flow.dispatch @torch_jit$async_dispatch_35::@torch_jit$async_dispatch_35_conv_14x14x576x3x3_f32(%67, %__constant_tensor_576x3x3xf32_27, %__constant_tensor_576xf32_54) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x16x16xf32>, tensor<576x3x3xf32>, tensor<576xf32>) -> tensor<576x196xf32, #encoding31>
    %69 = flow.tensor.encode %65 : tensor<96x196xf32> -> tensor<96x196xf32, #encoding32>
%70 = flow.dispatch @torch_jit$async_dispatch_36::@torch_jit$async_dispatch_36_matmul_like_96x196x576_f32(%68, %__hoisted_tensor_96x576xf32__encoded_10, %69, %__hoisted_tensor_96xf32__encoded_11) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x196xf32, #encoding31>, tensor<96x576xf32, #encoding8>, tensor<96x196xf32, #encoding32>, tensor<96xf32, #encoding50>) -> tensor<96x196xf32>
    %71 = flow.tensor.reshape %70 : tensor<96x196xf32> -> tensor<96x14x14xf32>
%72 = flow.dispatch @torch_jit$async_dispatch_34::@torch_jit$async_dispatch_34_matmul_like_576x14x14x96_f32(%71, %__constant_tensor_576x96xf32, %__constant_tensor_576xf32_55, %61) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<96x14x14xf32>, tensor<576x96xf32>, tensor<576xf32>, tensor<576x16x16xf32>) -> %61
%73 = flow.dispatch @torch_jit$async_dispatch_41::@torch_jit$async_dispatch_41_conv_7x7x576x3x3_f32(%72, %__constant_tensor_576x3x3xf32, %__constant_tensor_576xf32_56) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x16x16xf32>, tensor<576x3x3xf32>, tensor<576xf32>) -> tensor<576x49xf32, #encoding33>
%74 = flow.dispatch @torch_jit$async_dispatch_42::@torch_jit$async_dispatch_42_matmul_like_160x49x576_f32(%73, %__hoisted_tensor_160x576xf32__encoded, %__hoisted_tensor_160xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<576x49xf32, #encoding33>, tensor<160x576xf32, #encoding9>, tensor<160xf32, #encoding51>) -> tensor<160x49xf32>
    %75 = flow.tensor.reshape %74 : tensor<160x49xf32> -> tensor<160x7x7xf32>
    %76 = flow.tensor.splat %cst : tensor<960x9x9xf32>
%77 = flow.dispatch @torch_jit$async_dispatch_43::@torch_jit$async_dispatch_43_matmul_like_960x7x7x160_f32(%75, %__constant_tensor_960x160xf32_16, %__constant_tensor_960xf32, %76) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<160x7x7xf32>, tensor<960x160xf32>, tensor<960xf32>, tensor<960x9x9xf32>) -> %76
%78 = flow.dispatch @torch_jit$async_dispatch_44::@torch_jit$async_dispatch_44_conv_7x7x960x3x3_f32(%77, %__constant_tensor_960x3x3xf32_26, %__constant_tensor_960xf32_57) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x9x9xf32>, tensor<960x3x3xf32>, tensor<960xf32>) -> tensor<960x49xf32, #encoding35>
    %79 = flow.tensor.encode %74 : tensor<160x49xf32> -> tensor<160x49xf32, #encoding36>
%80 = flow.dispatch @torch_jit$async_dispatch_45::@torch_jit$async_dispatch_45_matmul_like_160x49x960_f32(%78, %__hoisted_tensor_160x960xf32__encoded, %79, %__hoisted_tensor_160xf32__encoded_12) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x49xf32, #encoding35>, tensor<160x960xf32, #encoding10>, tensor<160x49xf32, #encoding36>, tensor<160xf32, #encoding52>) -> tensor<160x49xf32>
    %81 = flow.tensor.reshape %80 : tensor<160x49xf32> -> tensor<160x7x7xf32>
%82 = flow.dispatch @torch_jit$async_dispatch_43::@torch_jit$async_dispatch_43_matmul_like_960x7x7x160_f32(%81, %__constant_tensor_960x160xf32_15, %__constant_tensor_960xf32_58, %76) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<160x7x7xf32>, tensor<960x160xf32>, tensor<960xf32>, tensor<960x9x9xf32>) -> %76
%83 = flow.dispatch @torch_jit$async_dispatch_44::@torch_jit$async_dispatch_44_conv_7x7x960x3x3_f32(%82, %__constant_tensor_960x3x3xf32_25, %__constant_tensor_960xf32_59) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x9x9xf32>, tensor<960x3x3xf32>, tensor<960xf32>) -> tensor<960x49xf32, #encoding35>
    %84 = flow.tensor.encode %80 : tensor<160x49xf32> -> tensor<160x49xf32, #encoding36>
%85 = flow.dispatch @torch_jit$async_dispatch_45::@torch_jit$async_dispatch_45_matmul_like_160x49x960_f32(%83, %__hoisted_tensor_160x960xf32__encoded_13, %84, %__hoisted_tensor_160xf32__encoded_14) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x49xf32, #encoding35>, tensor<160x960xf32, #encoding10>, tensor<160x49xf32, #encoding36>, tensor<160xf32, #encoding52>) -> tensor<160x49xf32>
    %86 = flow.tensor.reshape %85 : tensor<160x49xf32> -> tensor<160x7x7xf32>
%87 = flow.dispatch @torch_jit$async_dispatch_43::@torch_jit$async_dispatch_43_matmul_like_960x7x7x160_f32(%86, %__constant_tensor_960x160xf32, %__constant_tensor_960xf32_60, %76) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<160x7x7xf32>, tensor<960x160xf32>, tensor<960xf32>, tensor<960x9x9xf32>) -> %76
%88 = flow.dispatch @torch_jit$async_dispatch_50::@torch_jit$async_dispatch_50_conv_7x7x960x3x3_f32(%87, %__constant_tensor_960x3x3xf32, %__constant_tensor_960xf32_61) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x9x9xf32>, tensor<960x3x3xf32>, tensor<960xf32>) -> tensor<960x49xf32, #encoding37>
%89 = flow.dispatch @torch_jit$async_dispatch_51::@torch_jit$async_dispatch_51_matmul_like_320x49x960_f32(%88, %__hoisted_tensor_320x960xf32__encoded, %__hoisted_tensor_320xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<960x49xf32, #encoding37>, tensor<320x960xf32, #encoding11>, tensor<320xf32, #encoding53>) -> tensor<320x49xf32, #encoding38>
%90 = flow.dispatch @torch_jit$async_dispatch_52::@torch_jit$async_dispatch_52_matmul_like_1280x49x320_f32(%89, %__hoisted_tensor_1280x320xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<320x49xf32, #encoding38>, tensor<1280x320xf32, #encoding12>) -> tensor<1280x49xf32>
%91 = flow.dispatch @torch_jit$async_dispatch_53::@torch_jit$async_dispatch_53_reduction_1280x49_f32(%90, %__constant_tensor_1280xf32) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<1280x49xf32>, tensor<1280xf32>) -> tensor<1280xf32, #encoding41>
%92 = flow.dispatch @torch_jit$async_dispatch_54::@torch_jit$async_dispatch_54_matvec_like_1000x1280_f32(%91, %__hoisted_tensor_1000x1280xf32__encoded, %__hoisted_tensor_1000xf32__encoded) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<1280xf32, #encoding41>, tensor<1000x1280xf32, #encoding13>, tensor<1000xf32, #encoding14>) -> tensor<1000xf32>
    %93 = flow.tensor.reshape %92 : tensor<1000xf32> -> tensor<1x1000xf32>
    %94 = hal.tensor.barrier join(%93 : tensor<1x1000xf32>) => %arg2 : !hal.fence
    %95 = hal.tensor.export %94 : tensor<1x1000xf32> -> !hal.buffer_view
    util.return %95 : !hal.buffer_view
  }
  util.func public @torch_jit(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_a = util.global.load @device_a : !hal.device
    %fence = hal.fence.create device(%device_a : !hal.device) flags("None") : !hal.fence
    %1 = util.call @torch_jit$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}
