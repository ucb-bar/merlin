// Standard builtin headers
builtin.module {

  // =========================================================
  // SECTION 0: Topology Globals
  // We must define the device globals so we can reference them 
  // in the affinity attributes (#hal.device.promise<@name>).
  // =========================================================
  util.global private @device_a : !hal.device
  util.global private @device_b : !hal.device
  util.global private @device_ab : !hal.device

  // =========================================================
  // SECTION 1: The Executables (Kernels)
  // =========================================================

  // --- Model A (The Heavy Lifter) ---
  stream.executable private @model_a_exe {
    stream.executable.export public @layer_heavy_1 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    stream.executable.export public @layer_light_2 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }

    builtin.module {
      // Signature: Input Binding -> Output Binding
      // Note: In stream executable bodies, we work with !stream.binding
      func.func @layer_heavy_1(%input: !stream.binding, %output: !stream.binding) {
        return 
      }
      func.func @layer_light_2(%input: !stream.binding, %output: !stream.binding) {
        return 
      }
    }
  }

  // --- Model B (The Background Task) ---
  stream.executable private @model_b_exe {
    stream.executable.export public @layer_medium_1 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }

    builtin.module {
      func.func @layer_medium_1(%input: !stream.binding, %output: !stream.binding) {
        return 
      }
    }
  }

  // =========================================================
  // SECTION 2: The Scheduler
  // =========================================================

  func.func @static_deterministic_schedule(
        // Inputs are external resources (already in memory)
        %input_a: !stream.resource<external>,
        %input_b: !stream.resource<external>
    ) -> !stream.timepoint {

    // Define constants for resource sizes (Mandatory in Stream)
    // In a real scenario, these match your tensor byte sizes (e.g., 4xf32 = 16 bytes)
    %sz_input = arith.constant 1024 : index
    %sz_heavy_out = arith.constant 2048 : index
    %sz_final = arith.constant 1024 : index

    // 1. Anchor the Timeline (T=0)
    %t0 = stream.timepoint.immediate => !stream.timepoint

    // +++++++++++++++++++++++++++++++++++++++++++++
    // TIME STEP 1: Exclusive Heavy Compute
    // Device: AB (Cores 0+1)
    // +++++++++++++++++++++++++++++++++++++++++++++
    
    // We explicitly allocate the result memory for this layer.
    // 'transient' means it lives only for this graph execution.
    %res_a1_alloc, %t_alloc_a1 = stream.resource.alloca uninitialized
        on(#hal.device.promise<@device_ab>)
        await(%t0)
        => !stream.resource<transient>{%sz_heavy_out} => !stream.timepoint

    // Execute the heavy kernel
    %res_a1, %t1 = stream.async.execute
        await(%t_alloc_a1)             // Wait for allocation
        with(%input_a as %in: !stream.resource<external>{%sz_input}, 
             %res_a1_alloc as %out: !stream.resource<transient>{%sz_heavy_out})
        on(#hal.device.promise<@device_ab>) // <--- PINNED TO CORE 0+1
    {
        // Dispatch writes into %out
        %result = stream.async.dispatch @model_a_exe::@layer_heavy_1(%in, %out) : 
                  (!stream.resource<external>{%sz_input}, !stream.resource<transient>{%sz_heavy_out}) 
                  -> !stream.resource<transient>{%sz_heavy_out}
        stream.yield %result : !stream.resource<transient>{%sz_heavy_out}
    } => !stream.timepoint

    // +++++++++++++++++++++++++++++++++++++++++++++
    // TIME STEP 2: The Concurrent Fork
    // +++++++++++++++++++++++++++++++++++++++++++++

    // --- Path A: Model A Layer 2 on Device A (Core 0) ---
    
    // Allocation for A2 result
    %res_a2_alloc, %t_alloc_a2 = stream.resource.alloca uninitialized
        on(#hal.device.promise<@device_a>)
        await(%t0) // Allocation can happen early, but execution waits for T1
        => !stream.resource<transient>{%sz_final} => !stream.timepoint

    // Join timepoints for A2 execution: 
    // 1. Heavy layer must be done (%t1)
    // 2. Output buffer must be ready (%t_alloc_a2)
    %t_ready_a2 = stream.timepoint.join max(%t1, %t_alloc_a2) => !stream.timepoint

    %res_a2, %t2_a = stream.async.execute
        await(%t_ready_a2)
        with(%res_a1 as %in: !stream.resource<transient>{%sz_heavy_out},
             %res_a2_alloc as %out: !stream.resource<transient>{%sz_final})
        on(#hal.device.promise<@device_a>)  // <--- PINNED TO CORE 0
    {
        %result = stream.async.dispatch @model_a_exe::@layer_light_2(%in, %out) : 
                  (!stream.resource<transient>{%sz_heavy_out}, !stream.resource<transient>{%sz_final}) 
                  -> !stream.resource<transient>{%sz_final}
        stream.yield %result : !stream.resource<transient>{%sz_final}
    } => !stream.timepoint

    // --- Path B: Model B Layer 1 on Device B (Core 1) ---
    
    // Allocation for B1 result
    %res_b1_alloc, %t_alloc_b1 = stream.resource.alloca uninitialized
        on(#hal.device.promise<@device_b>)
        await(%t0) 
        => !stream.resource<transient>{%sz_final} => !stream.timepoint

    // CRITICAL SCHEDULING LOGIC:
    // Model B depends on %t1 (End of Heavy Layer).
    // NOT because it needs data from A, but because Core 1 was busy in Device AB!
    %t_ready_b1 = stream.timepoint.join max(%t1, %t_alloc_b1) => !stream.timepoint

    %res_b1, %t2_b = stream.async.execute
        await(%t_ready_b1)
        with(%input_b as %in: !stream.resource<external>{%sz_input},
             %res_b1_alloc as %out: !stream.resource<transient>{%sz_final})
        on(#hal.device.promise<@device_b>)  // <--- PINNED TO CORE 1
    {
        %result = stream.async.dispatch @model_b_exe::@layer_medium_1(%in, %out) : 
                  (!stream.resource<external>{%sz_input}, !stream.resource<transient>{%sz_final}) 
                  -> !stream.resource<transient>{%sz_final}
        stream.yield %result : !stream.resource<transient>{%sz_final}
    } => !stream.timepoint

    // +++++++++++++++++++++++++++++++++++++++++++++
    // TIME STEP 3: The Join
    // +++++++++++++++++++++++++++++++++++++++++++++
    
    // Wait for both concurrent branches to finish
    %t_final = stream.timepoint.join max(%t2_a, %t2_b) => !stream.timepoint
    
    // (Optional) Here you would export/return the results, but for now we return the sync token.
    return %t_final : !stream.timepoint
  }
}