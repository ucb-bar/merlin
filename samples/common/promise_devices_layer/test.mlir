module attributes {
  // Define the topology at the top level
  stream.topology = #hal.device.topology<
    links = [
      // 1. Link A <-> AB (Unified Memory)
      // Use the symbol name directly: @device_a
      (@device_a -> @device_ab = {transparent_access = true, unified_memory = true}),
      (@device_ab -> @device_a = {transparent_access = true, unified_memory = true}),
      
      // 2. Link B <-> AB (Unified Memory)
      (@device_b -> @device_ab = {transparent_access = true, unified_memory = true}),
      (@device_ab -> @device_b = {transparent_access = true, unified_memory = true}),

      // 3. OPTIONAL: Link A <-> B (Unified Memory)
      // Since they are both on the same CPU RAM, you should link them too!
      // This allows direct transfer from A to B without going through AB logically.
      //(@device_a -> @device_b = {transparent_access = true, unified_memory = true}),
      //(@device_b -> @device_a = {transparent_access = true, unified_memory = true})
    ]
  >
} {
  func.func @main(
    // Input: Starts on Device A (Core 0)
    %input: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (
    // Output: Return on Device A
    tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) {
    // ---------------------------------------------------------
    // STAGE 1: Device A (Core 0)
    // ---------------------------------------------------------
    %c1 = arith.constant dense<1.0> : tensor<4xf32>
    %res_a = arith.addf %input, %c1 : tensor<4xf32>
    
    // ---------------------------------------------------------
    // STAGE 2: Transfer Device A -> Device B (Core 1)
    // ---------------------------------------------------------
    // The topology tells the compiler this is a "Zero-Copy" view change, 
    // rather than a full malloc+memcpy.
    %input_b = flow.tensor.transfer %res_a : tensor<4xf32> to #hal.device.promise<@device_b>
    
    %c2 = arith.constant dense<2.0> : tensor<4xf32>
    %res_b = arith.mulf %input_b, %c2 : tensor<4xf32>

    // ---------------------------------------------------------
    // STAGE 3: Transfer Device B -> Device AB (Cluster 0+1)
    // ---------------------------------------------------------
    %input_ab = flow.tensor.transfer %res_b : tensor<4xf32> to #hal.device.promise<@device_ab>

    %c10 = arith.constant dense<10.0> : tensor<4xf32>
    %res_ab = arith.addf %input_ab, %c10 : tensor<4xf32>

    // ---------------------------------------------------------
    // STAGE 4: Transfer back to Device A (Core 0)
    // ---------------------------------------------------------
    %input_final = flow.tensor.transfer %res_ab : tensor<4xf32> to #hal.device.promise<@device_a>
    
    return %input_final : tensor<4xf32> 
  }
}