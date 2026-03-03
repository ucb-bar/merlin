func.func @schedule(
    %input: tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (
    tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    tensor<128x128xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) {
    // Constants
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant dense<1.0> : tensor<128x128xf32>
    %c2 = arith.constant dense<2.0> : tensor<128x128xf32>
    
    // Init for Matmul
    %empty = tensor.empty() : tensor<128x128xf32>
    %init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>

    // =========================================================
    // TIMEPOINT 1: Parallel Divergence
    // Core A: X1 (Heavy, ~2 timesteps)
    // Core B: Y1, then Z1 (Light, fit inside X1's time)
    // =========================================================

    // --- LANE A ---
    %in_a = flow.tensor.transfer %input : tensor<128x128xf32> to #hal.device.promise<@device_a>
    // [X1] Heavy Compute
    %x1 = linalg.matmul 
      ins(%in_a, %in_a : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

    // --- LANE B ---
    %in_b = flow.tensor.transfer %input : tensor<128x128xf32> to #hal.device.promise<@device_b>
    
    // [Y1] First op on B
    %y1 = arith.addf %in_b, %c1 : tensor<128x128xf32>
    
    // [Z1] Second op on B (Independent data, but scheduled after Y1 on this core)
    %z1 = arith.mulf %in_b, %c2 : tensor<128x128xf32>

    // =========================================================
    // TIMEPOINT 2: Shared Block "Stretched Wide"
    // Execution merges onto Device AB for X2
    // =========================================================

    // Transfer X1 to the shared device
    %x1_ab = flow.tensor.transfer %x1 : tensor<128x128xf32> to #hal.device.promise<@device_ab>

    // [X2] Runs on AB. (Y and Z must wait because AB consumes B)
    %x2 = arith.subf %x1_ab, %c1 : tensor<128x128xf32>

    // =========================================================
    // TIMEPOINT 3: Parallel Split
    // Core A: X3
    // Core B: Y2 (Resuming Y stream)
    // =========================================================

    // --- LANE A ---
    // Transfer X2 result back to A for X3
    %x2_a = flow.tensor.transfer %x2 : tensor<128x128xf32> to #hal.device.promise<@device_a>
    // [X3]
    %x3 = arith.divf %x2_a, %c2 : tensor<128x128xf32>

    // --- LANE B ---
    // Y stream resumes here using Y1 result. 
    // (Ideally Y1 is still on B, or we transfer it back if context was lost)
    %y1_b = flow.tensor.transfer %y1 : tensor<128x128xf32> to #hal.device.promise<@device_b>
    // [Y2]
    %y2 = arith.mulf %y1_b, %c2 : tensor<128x128xf32>

    // =========================================================
    // TIMEPOINT 4: Final Shared Block
    // Execution merges onto Device AB for Z2
    // =========================================================

    // Transfer Z1 (from Timepoint 1) to AB
    %z1_ab = flow.tensor.transfer %z1 : tensor<128x128xf32> to #hal.device.promise<@device_ab>

    // [Z2] Final "Stretched" op
    %z2 = arith.addf %z1_ab, %c1 : tensor<128x128xf32>

    // =========================================================
    // RETURN
    // Collect all results back to A
    // =========================================================
    
    %final_y = flow.tensor.transfer %y2 : tensor<128x128xf32> to #hal.device.promise<@device_a>
    %final_z = flow.tensor.transfer %z2 : tensor<128x128xf32> to #hal.device.promise<@device_a>

    return %x3, %final_y, %final_z : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
}