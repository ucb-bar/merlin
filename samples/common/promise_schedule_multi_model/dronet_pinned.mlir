#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0) -> (d0)>
#device_target_primary = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_embedded_elf_x86_64]> : !hal.device
#device_target_secondary = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_embedded_elf_x86_64]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_primary : !hal.device
  util.global private @device_b = #device_target_secondary : !hal.device
  flow.executable private @schedule_dispatch_0 {
    flow.executable.export public @schedule_dispatch_0_pack_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_0_pack_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x128x8x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = tensor.empty() : tensor<16x128x8x1xf32>
        %pack = linalg.pack %0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %1 : tensor<128x128xf32> -> tensor<16x128x8x1xf32>
        iree_tensor_ext.dispatch.tensor.store %pack, %arg1, offsets = [0, 0, 0, 0], sizes = [16, 128, 8, 1], strides = [1, 1, 1, 1] : tensor<16x128x8x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x128x8x1xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_1 {
    flow.executable.export public @schedule_dispatch_1_pack_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_1_pack_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x128x4x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = tensor.empty() : tensor<32x128x4x1xf32>
        %pack = linalg.pack %0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 1] into %1 : tensor<128x128xf32> -> tensor<32x128x4x1xf32>
        iree_tensor_ext.dispatch.tensor.store %pack, %arg1, offsets = [0, 0, 0, 0], sizes = [32, 128, 4, 1], strides = [1, 1, 1, 1] : tensor<32x128x4x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x128x4x1xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_2 {
    flow.executable.export public @schedule_dispatch_2_mmt4d_16x32x128x8x4x1_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_2_mmt4d_16x32x128x8x4x1_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x128x8x1xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128x4x1xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32x8x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 128, 8, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x128x8x1xf32>> -> tensor<16x128x8x1xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [32, 128, 4, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128x4x1xf32>> -> tensor<32x128x4x1xf32>
        %2 = tensor.empty() : tensor<16x32x8x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x32x8x4xf32>) -> tensor<16x32x8x4xf32>
        %4 = linalg.mmt4d ins(%0, %1 : tensor<16x128x8x1xf32>, tensor<32x128x4x1xf32>) outs(%3 : tensor<16x32x8x4xf32>) -> tensor<16x32x8x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [16, 32, 8, 4], strides = [1, 1, 1, 1] : tensor<16x32x8x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32x8x4xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_3 {
    flow.executable.export public @schedule_dispatch_3_unpack_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_3_unpack_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x32x8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 32, 8, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x32x8x4xf32>> -> tensor<16x32x8x4xf32>
        %1 = tensor.empty() : tensor<128x128xf32>
        %unpack = linalg.unpack %0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %1 : tensor<16x32x8x4xf32> -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %unpack, %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_4 {
    flow.executable.export public @schedule_dispatch_4_elementwise_16384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_4_elementwise_16384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 2.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.mulf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_5 {
    flow.executable.export public @schedule_dispatch_5_elementwise_16384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_5_elementwise_16384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 1.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.subf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_6 {
    flow.executable.export public @schedule_dispatch_6_elementwise_16384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_6_elementwise_16384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 2.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.divf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_7 {
    flow.executable.export public @schedule_dispatch_7_elementwise_16384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_7_elementwise_16384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 1.000000e+00 : f32
        %cst_0 = arith.constant 2.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.addf %in, %cst : f32
          %4 = arith.mulf %3, %cst_0 : f32
          linalg.yield %4 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @schedule_dispatch_8 {
    flow.executable.export public @schedule_dispatch_8_elementwise_16384_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @schedule_dispatch_8_elementwise_16384_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 1.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.addf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  util.func public @schedule(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @schedule(%input0: tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}) -> (%output0: tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}, %output1: tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}, %output2: tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>})", iree.abi.model = "coarse-fences"}} {
    %0 = hal.tensor.import on(#hal.device.affinity<@device_a>) wait(%arg1) => %arg0 "input0" : !hal.buffer_view -> tensor<128x128xf32>
    %1 = flow.tensor.transfer %0 : tensor<128x128xf32> to #hal.device.affinity<@device_a>
%2 = flow.dispatch @schedule_dispatch_0::@schedule_dispatch_0_pack_f32(%1) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<128x128xf32>) -> tensor<16x128x8x1xf32>
%3 = flow.dispatch @schedule_dispatch_1::@schedule_dispatch_1_pack_f32(%1) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<128x128xf32>) -> tensor<32x128x4x1xf32>
%4 = flow.dispatch @schedule_dispatch_2::@schedule_dispatch_2_mmt4d_16x32x128x8x4x1_f32(%2, %3) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16x128x8x1xf32>, tensor<32x128x4x1xf32>) -> tensor<16x32x8x4xf32>
%5 = flow.dispatch @schedule_dispatch_3::@schedule_dispatch_3_unpack_f32(%4) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16x32x8x4xf32>) -> tensor<128x128xf32>
    %6 = flow.tensor.transfer %0 : tensor<128x128xf32> to #hal.device.affinity<@device_b>
    %7 = flow.tensor.reshape %6 : tensor<128x128xf32> -> tensor<16384xf32>
%8 = flow.dispatch @schedule_dispatch_4::@schedule_dispatch_4_elementwise_16384_f32(%7) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16384xf32>) -> tensor<16384xf32>
    %9 = flow.tensor.transfer %5 : tensor<128x128xf32> to #hal.device.affinity<@device_b>
    %10 = flow.tensor.reshape %9 : tensor<128x128xf32> -> tensor<16384xf32>
%11 = flow.dispatch @schedule_dispatch_5::@schedule_dispatch_5_elementwise_16384_f32(%10) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16384xf32>) -> tensor<16384xf32>
    %12 = flow.tensor.transfer %11 : tensor<16384xf32> to #hal.device.affinity<@device_a>
%13 = flow.dispatch @schedule_dispatch_6::@schedule_dispatch_6_elementwise_16384_f32(%12) { stream.affinity = #hal.device.affinity<@device_b> } : (tensor<16384xf32>) -> tensor<16384xf32>
    %14 = flow.tensor.reshape %13 : tensor<16384xf32> -> tensor<128x128xf32>
%15 = flow.dispatch @schedule_dispatch_7::@schedule_dispatch_7_elementwise_16384_f32(%7) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16384xf32>) -> tensor<16384xf32>
    %16 = flow.tensor.transfer %8 : tensor<16384xf32> to #hal.device.affinity<@device_b>
%17 = flow.dispatch @schedule_dispatch_8::@schedule_dispatch_8_elementwise_16384_f32(%16) { stream.affinity = #hal.device.affinity<@device_a> } : (tensor<16384xf32>) -> tensor<16384xf32>
    %18 = flow.tensor.transfer %15 : tensor<16384xf32> to #hal.device.affinity<@device_a>
    %19 = flow.tensor.reshape %18 : tensor<16384xf32> -> tensor<128x128xf32>
    %20 = flow.tensor.transfer %17 : tensor<16384xf32> to #hal.device.affinity<@device_a>
    %21 = flow.tensor.reshape %20 : tensor<16384xf32> -> tensor<128x128xf32>
    %22:3 = hal.tensor.barrier join(%14, %19, %21 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) => %arg2 : !hal.fence
    %23 = hal.tensor.export on(#hal.device.affinity<@device_a>) %22#0 "output0" : tensor<128x128xf32> -> !hal.buffer_view
    %24 = hal.tensor.export on(#hal.device.affinity<@device_a>) %22#1 "output1" : tensor<128x128xf32> -> !hal.buffer_view
    %25 = hal.tensor.export on(#hal.device.affinity<@device_a>) %22#2 "output2" : tensor<128x128xf32> -> !hal.buffer_view
    util.return %23, %24, %25 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}
