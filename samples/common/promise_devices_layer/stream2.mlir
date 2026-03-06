#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver3", cpu_features = "+prfchw,-cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,-amx-fp8,+xsaves,-avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,-avx10.1,-avx512ifma,+xsave,+sse4.2,-tsxldtrk,-sm3,-ptwrite,-widekl,-movrs,+invpcid,+64bit,+xsavec,-avx512vpopcntdq,+cmov,-avx512vp2intersect,-avx512cd,+movbe,-avxvnniint8,-ccmp,-amx-int8,-kl,-sha512,-avxvnni,-rtm,+adx,+avx2,-hreset,-movdiri,-serialize,+vpclmulqdq,-avx512vl,-uintr,-cf,+clflushopt,-raoint,-cmpccxadd,+bmi,-amx-tile,+sse,-gfni,-avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,-avx512f,-amx-bf16,-avx512bf16,-avx512vnni,-push2pop2,+cx8,-avx512bw,+sse3,+pku,-nf,-amx-tf32,-amx-avx512,+fsgsbase,+clzero,+mwaitx,-lwp,+lzcnt,+sha,-movdir64b,-ppx,+wbnoinvd,-enqcmd,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+cx16,-avx10.2,+bmi2,+fma,+popcnt,-avxifma,+f16c,-avx512bitalg,+rdpru,+clwb,+mmx,+sse2,+rdseed,-avx512vbmi2,-prefetchi,-amx-movrs,+rdpid,-fma4,-avx512vbmi,+shstk,+vaes,-waitpkg,-sgx,+fxsr,-avx512dq,+sse4a", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 32 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0) -> (d0)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_embedded_elf_x86_64]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_embedded_elf_x86_64]> : !hal.device
#device_target_local_2_ = #hal.device.target<"local", {ordinal = 2 : index}, [#executable_target_embedded_elf_x86_64]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_ab = #device_target_local_2_
  util.global private @device_b = #device_target_local_1_

  // =========================================================
  // 2. KERNEL DEFINITIONS
  // =========================================================

  // --- KERNEL 1: Wide (Matmul) ---
  stream.executable private @kernels_wide {
    stream.executable.export public @matmul_f32 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @matmul_f32(%lhs: !stream.binding, %rhs: !stream.binding, %out: !stream.binding) {
        %c0 = arith.constant 0 : index
        %lhs_bind = stream.binding.subspan %lhs[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %rhs_bind = stream.binding.subspan %rhs[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %out_bind = stream.binding.subspan %out[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        
        %empty = tensor.empty() : tensor<128x128xf32>
        %cst_0 = arith.constant 0.0 : f32
        %init = linalg.fill ins(%cst_0 : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        %res = linalg.matmul ins(%lhs_t, %rhs_t : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
        
        iree_tensor_ext.dispatch.tensor.store %res, %out_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }

  // --- KERNEL 2: Short (Add) ---
  stream.executable private @kernels_short {
    stream.executable.export public @add_f32 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @add_f32(%input: !stream.binding, %out: !stream.binding) {
        %c0 = arith.constant 0 : index
        %in_bind = stream.binding.subspan %input[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %out_bind = stream.binding.subspan %out[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        
        %in_t = iree_tensor_ext.dispatch.tensor.load %in_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %empty = tensor.empty() : tensor<128x128xf32>
        %cst_1 = arith.constant 1.0 : f32
        
        %res = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%in_t : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) {
        ^bb0(%in: f32, %out_elem: f32):
          %val = arith.addf %in, %cst_1 : f32
          linalg.yield %val : f32
        } -> tensor<128x128xf32>

        iree_tensor_ext.dispatch.tensor.store %res, %out_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }

  // --- KERNEL 3: Tall (Mul) ---
  stream.executable private @kernels_tall {
    stream.executable.export public @mul_f32 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @mul_f32(%in_a: !stream.binding, %in_b: !stream.binding, %out: !stream.binding) {
        %c0 = arith.constant 0 : index
        %a_bind = stream.binding.subspan %in_a[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %b_bind = stream.binding.subspan %in_b[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
        %out_bind = stream.binding.subspan %out[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        
        %a_t = iree_tensor_ext.dispatch.tensor.load %a_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %b_t = iree_tensor_ext.dispatch.tensor.load %b_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        
        %empty = tensor.empty() : tensor<128x128xf32>
        
        %res = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%a_t, %b_t : tensor<128x128xf32>, tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) {
        ^bb0(%i1: f32, %i2: f32, %o: f32):
          %val = arith.mulf %i1, %i2 : f32
          linalg.yield %val : f32
        } -> tensor<128x128xf32>

        iree_tensor_ext.dispatch.tensor.store %res, %out_bind, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }

  // =========================================================
  // 3. THE STATIC SCHEDULER
  // =========================================================
  util.func @run_static_schedule(%input: !stream.resource<external>) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %sz = arith.constant 65536 : index
    %t0 = stream.timepoint.immediate => !stream.timepoint

    // -------------------------------------------------------------------------
    // PHASE 1: [X1] (Wide) vs [Y1] (Short)
    // -------------------------------------------------------------------------

    // [X1] on Device A (Core 0)
    // Note: Changed `promise` to `affinity` to ensure static compiler resolution
    %mem_x1, %t_alloc_x1 = stream.resource.alloca uninitialized 
        on(#hal.device.affinity<@device_a>) 
        await(%t0) => 
        !stream.resource<transient>{%sz} => !stream.timepoint
    
    %res_x1, %tp_x1_done = stream.async.execute
        on(#hal.device.affinity<@device_a>)
        await(%t_alloc_x1) =>
        with(%input as %in: !stream.resource<external>{%sz}, %mem_x1 as %out: !stream.resource<transient>{%sz})
        -> !stream.resource<transient>{%sz}
    {
      %1 = stream.async.dispatch @kernels_wide::@matmul_f32(%in[%c0 to %sz for %sz], %in[%c0 to %sz for %sz], %out[%c0 to %sz for %sz]) : (!stream.resource<external>{%sz}, !stream.resource<external>{%sz}, !stream.resource<transient>{%sz}) -> !stream.resource<transient>{%sz}
      stream.yield %1 : !stream.resource<transient>{%sz}
    } => !stream.timepoint

    // [Y1] on Device B (Core 1)
    %mem_y1, %t_alloc_y1 = stream.resource.alloca uninitialized 
        on(#hal.device.affinity<@device_b>) 
        await(%t0) => 
        !stream.resource<transient>{%sz} => !stream.timepoint

    %res_y1, %tp_y1_done = stream.async.execute
        on(#hal.device.affinity<@device_b>)
        await(%t_alloc_y1) =>
        with(%input as %in: !stream.resource<external>{%sz}, %mem_y1 as %out: !stream.resource<transient>{%sz})
        -> !stream.resource<transient>{%sz}
    {
      %1 = stream.async.dispatch @kernels_short::@add_f32(%in[%c0 to %sz for %sz], %out[%c0 to %sz for %sz]) : (!stream.resource<external>{%sz}, !stream.resource<transient>{%sz}) -> !stream.resource<transient>{%sz}
      stream.yield %1 : !stream.resource<transient>{%sz}
    } => !stream.timepoint

    // -------------------------------------------------------------------------
    // PHASE 2: Gap Filling [Z1] on Device B
    // -------------------------------------------------------------------------

    // [Z1] on Device B (Core 1)
    %mem_z1, %t_alloc_z1 = stream.resource.alloca uninitialized 
        on(#hal.device.affinity<@device_b>) 
        await(%t0) => 
        !stream.resource<transient>{%sz} => !stream.timepoint
    
    // DEPENDENCY: Wait for Y1
    %tp_z1_ready = stream.timepoint.join max(%tp_y1_done, %t_alloc_z1) => !stream.timepoint

    %res_z1, %tp_z1_done = stream.async.execute
        on(#hal.device.affinity<@device_b>)
        await(%tp_z1_ready) =>
        with(%res_y1 as %in: !stream.resource<transient>{%sz}, %mem_z1 as %out: !stream.resource<transient>{%sz})
        -> !stream.resource<transient>{%sz}
    {
      %1 = stream.async.dispatch @kernels_short::@add_f32(%in[%c0 to %sz for %sz], %out[%c0 to %sz for %sz]) : (!stream.resource<transient>{%sz}, !stream.resource<transient>{%sz}) -> !stream.resource<transient>{%sz}
      stream.yield %1 : !stream.resource<transient>{%sz}
    } => !stream.timepoint

    // -------------------------------------------------------------------------
    // PHASE 3: Catch Up [X2] on Device A
    // -------------------------------------------------------------------------

    // [X2] on Device A (Core 0)
    %mem_x2, %t_alloc_x2 = stream.resource.alloca uninitialized 
        on(#hal.device.affinity<@device_a>) 
        await(%t0) => 
        !stream.resource<transient>{%sz} => !stream.timepoint

    // DEPENDENCY: Wait for X1
    %tp_x2_ready = stream.timepoint.join max(%tp_x1_done, %t_alloc_x2) => !stream.timepoint

    %res_x2, %tp_x2_done = stream.async.execute
        on(#hal.device.affinity<@device_a>)
        await(%tp_x2_ready) =>
        with(%res_x1 as %in: !stream.resource<transient>{%sz}, %mem_x2 as %out: !stream.resource<transient>{%sz})
        -> !stream.resource<transient>{%sz}
    {
      %1 = stream.async.dispatch @kernels_short::@add_f32(%in[%c0 to %sz for %sz], %out[%c0 to %sz for %sz]) : (!stream.resource<transient>{%sz}, !stream.resource<transient>{%sz}) -> !stream.resource<transient>{%sz}
      stream.yield %1 : !stream.resource<transient>{%sz}
    } => !stream.timepoint

    // -------------------------------------------------------------------------
    // PHASE 4: The "Tall" Op [Z2] on Device AB
    // -------------------------------------------------------------------------

    // [Z2] on Device AB (Core 0 + Core 1)
    %mem_z2, %t_alloc_z2 = stream.resource.alloca uninitialized 
        on(#hal.device.affinity<@device_ab>) 
        await(%t0) => 
        !stream.resource<transient>{%sz} => !stream.timepoint

    // SYNCHRONIZATION POINT: Join timelines from both devices.
    %tp_ab_ready = stream.timepoint.join max(%tp_x2_done, %tp_z1_done, %t_alloc_z2) => !stream.timepoint

    %res_z2, %tp_z2_done = stream.async.execute
        on(#hal.device.affinity<@device_ab>)
        await(%tp_ab_ready) =>
        with(%res_x2 as %in_a: !stream.resource<transient>{%sz}, 
             %res_z1 as %in_b: !stream.resource<transient>{%sz},
             %mem_z2 as %out: !stream.resource<transient>{%sz})
        -> !stream.resource<transient>{%sz}
    {
      %1 = stream.async.dispatch @kernels_tall::@mul_f32(%in_a[%c0 to %sz for %sz], %in_b[%c0 to %sz for %sz], %out[%c0 to %sz for %sz]) : (!stream.resource<transient>{%sz}, !stream.resource<transient>{%sz}, !stream.resource<transient>{%sz}) -> !stream.resource<transient>{%sz}
      stream.yield %1 : !stream.resource<transient>{%sz}
    } => !stream.timepoint

    return %tp_z2_done : !stream.timepoint
  }
}