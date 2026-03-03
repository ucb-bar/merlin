// File: qgemm_transform_spec.mlir
#map_1d = affine_map<(d0) -> (d0)>
#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_scalar_in_2d = affine_map<(d0, d1) -> ()>
#map_scalar_in_1d = affine_map<(d0) -> ()>
#map_broadcast_1d_in_2d = affine_map<(d0, d1) -> (d1)>

// 1. Define the target architecture (RISC-V 64-bit)
#riscv_64_target = #hal.executable.target<"llvm-cpu", "riscv_64", {
  target_triple = "riscv64-unknown-linux-gnu",
  cpu_features = "+m,+a,+f,+d,+v,+zvl512b,+zvbb1p0",
  target_abi = "lp64d",
  data_layout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
}>
// 2. Define the pipeline layout for the custom kernel
// Constants: 3 (M, N, K)
// Bindings: 4 (A, B, Bias_i32, C_out_i32)
#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>, // %A
  #hal.pipeline.binding<storage_buffer, ReadOnly>, // %B
  #hal.pipeline.binding<storage_buffer, ReadOnly>, // %Bias_i32
  #hal.pipeline.binding<storage_buffer>          // %C_out_i32
]>

module attributes {transform.with_named_sequence} {

  // 3. Define the executable linking to the pre-compiled object file
  hal.executable private @qgemm_executable {
    hal.executable.variant public @riscv_64 target(#riscv_64_target) objects([
      #hal.executable.object<{
        // This path must be discoverable by iree-compile via
        // --iree-hal-executable-object-search-path=
        path = "riscv_qgemm.o"
      }>
    ]) {
      // Export the wrapper function
      hal.executable.export public @qgemm_i8_bias_i32 ordinal(0)
      layout(#pipeline_layout)
      // This kernel does all work in one call, so dispatch (1, 1, 1) workgroups.
      count(%device: !hal.device, %workload_M: index, %workload_N: index, %workload_K: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }

      builtin.module {
        // 4. Import the C shim function
        // The signature here must match the memref ABI expansion of the shim
        func.func private @qgemm_i8_bias_i32_workgroup(
          %at: memref<1x?xi8>,   // A (M=1, K=?)
          %b: memref<?x?xi8>,  // B (K=?, N=?)
          %bias: memref<?xi32>, // Bias (N=?)
          %out: memref<1x?xi32>, // C_out (M=1, N=?)
          %M: index, %N: index, %K: index
        ) attributes {hal.import.static}

        // 5. Define the exported kernel wrapper
        // This function is called by the (1, 1, 1) workgroup.
        func.func @qgemm_i8_bias_i32() {
          %c0 = arith.constant 0 : index

          // Load M, N, K from push constants
          %M_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
          %N_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
          %K_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
          %M = arith.index_castui %M_i32 : i32 to index
          %N = arith.index_castui %N_i32 : i32 to index
          %K = arith.index_castui %K_i32 : i32 to index

          // Get memref subspans for all bindings
          %A_mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) : memref<1x?xi8>{%K}
          %B_mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0) : memref<?x?xi8>{%K, %N}
          %Bias_mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            alignment(64) offset(%c0) : memref<?xi32>{%N}
          %C_out_mem = hal.interface.binding.subspan layout(#pipeline_layout) binding(3)
            alignment(64) offset(%c0) : memref<1x?xi32>{%N}

          // Call the imported C shim function
          func.call @qgemm_i8_bias_i32_workgroup(
            %A_mem, %B_mem, %Bias_mem, %C_out_mem, %M, %N, %K
          ) : (memref<1x?xi8>, memref<?x?xi8>, memref<?xi32>, memref<1x?xi32>, index, index, index) -> ()

          return
        }
      }
    }
  }

  // 6. Define the replacement function
  // This function is what replaces the matched DAG
  util.func private @call_qgemm_dequant(
    %A: tensor<1x?xi8>,
    %B: tensor<?x?xi8>,
    %Bias_i32: tensor<?xi32>,
    %Scale: tensor<f32> // This is the shared dequant scale
  ) -> tensor<1x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Get dimensions
    %M_idx = tensor.dim %A, %c0 : tensor<1x?xi8> // M=1
    %K_idx = tensor.dim %A, %c1 : tensor<1x?xi8>
    %N_idx = tensor.dim %B, %c1 : tensor<?x?xi8>

    // Prepare an empty output tensor for the i32 result
    %C_i32_empty = tensor.empty(%N_idx) : tensor<1x?xi32>

    // Cast dims for push constants
    %M_i32 = arith.index_cast %M_idx : index to i32
    %N_i32 = arith.index_cast %N_idx : index to i32
    %K_i32 = arith.index_cast %K_idx : index to i32

    // Dispatch the custom kernel
    %C_i32_result = flow.dispatch @qgemm_executable::@executable_target_embedded_elf_riscv_64::@qgemm_i8_bias_i32
      [%M_idx, %N_idx, %K_idx] // Workload
      (%M_i32, %N_i32, %K_i32, %A, %B, %Bias_i32) // Constants + Inputs
      : (i32, i32, i32, tensor<1x?xi8>{%K_idx}, tensor<?x?xi8>{%K_idx, %N_idx}, tensor<?xi32>{%N_idx})
      -> tensor<1x?xi32>{%N_idx}

    // Now, perform the dequantization that was part of the matched pattern
    %Scale_f32 = tensor.extract %Scale[] : tensor<f32>
    %C_f32_empty = tensor.empty(%N_idx) : tensor<1x?xf32>
    %C_f32_result = linalg.generic {
      indexing_maps = [#map_2d, #map_scalar_in_2d, #map_2d],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%C_i32_result, %Scale_f32 : tensor<1x?xi32>, f32)
    outs(%C_f32_empty : tensor<1x?xf32>) {
    ^bb0(%in_i32: i32, %in_scale: f32, %out_f32: f32):
      %in_f32 = arith.sitofp %in_i32 : i32 to f32
      %scaled = arith.mulf %in_f32, %in_scale : f32
      linalg.yield %scaled : f32
    } -> tensor<1x?xf32>

    util.return %C_f32_result : tensor<1x?xf32>
  }

  // 7. Define the matcher
  transform.named_sequence @match_qgemm_bias_fused_scale(
    %root: !transform.any_op {transform.readonly}
  ) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%A: tensor<1x?xi8>, %B: tensor<?x?xi8>, %Bias_i32: tensor<?xi32>, 
          %Scale_Matmul_tensor: tensor<f32>, %Scale_Bias_tensor: tensor<f32>):
      
      //%c0 = arith.constant 0 : index
      //%c1 = arith.constant 1 : index
      //%dim_0 = tensor.dim %A, %c0 : tensor<1x?xi8> // M (which is 1)
      //%dim_K = tensor.dim %A, %c1 : tensor<1x?xi8> // K
      //%dim_1 = tensor.dim %B, %c1 : tensor<?x?xi8> // N

      %cst_i8_max = arith.constant -1.280000e+02 : f32
      %cst_i8_min = arith.constant 1.270000e+02 : f32

      %cst_f32_0 = arith.constant 0.000000e+00 : f32

      %cst_9 = arith.constant 0.0354968868 : f32

      
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%Bias_i32 : tensor<?xi32>) outs(%1 : tensor<?xf32>) {
      ^bb0(%in: i32, %out: f32):
        %44 = arith.sitofp %in : i32 to f32
        %45 = arith.mulf %44, %Scale_Bias_tensor : f32
        linalg.yield %45 : f32
      } -> tensor<?xf32>
      %5 = tensor.empty() : tensor<1x?xi8>
      %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1x?xf32>) outs(%5 : tensor<1x?xi8>) {
      ^bb0(%in: f32, %out: i8):
        %44 = arith.divf %in, %cst_9 : f32
        %45 = math.roundeven %44 : f32
        %46 = arith.addf %45, %cst_f32_0 : f32
        %47 = arith.maximumf %46, %cst_i8_max : f32
        %48 = arith.minimumf %47, %cst_i8_min : f32
        %49 = arith.fptosi %48 : f32 to i8
        linalg.yield %49 : i8
      } -> tensor<1x?xi8>
      %7 = tensor.empty() : tensor<?x?xi8>
      %transposed = linalg.transpose ins(%cst : tensor<?x?xi8>) outs(%7 : tensor<?x?xi8>) permutation = [1, 0] 
      %8 = tensor.empty() : tensor<1x?xi32>
      %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<1x?xi32>) -> tensor<1x?xi32>
      %10 = linalg.quantized_matmul ins(%6, %transposed, %c0_i32, %c0_i32 : tensor<1x?xi8>, tensor<?x?xi8>, i32, i32) outs(%9 : tensor<1x?xi32>) -> tensor<1x?xi32>
      %11 = tensor.empty() : tensor<1x?xf32>
      %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x?xi32>) outs(%11 : tensor<1x?xf32>) {
      ^bb0(%in: i32, %out: f32):
        %44 = arith.sitofp %in : i32 to f32
        %45 = arith.mulf %44, %cst_10 : f32
        linalg.yield %45 : f32
      } -> tensor<1x?xf32>
      %13 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%12, %2 : tensor<1x?xf32>, tensor<?xf32>) outs(%11 : tensor<1x?xf32>) {
      ^bb0(%in: f32, %in_16: f32, %out: f32):
        %44 = arith.addf %in, %in_16 : f32
        linalg.yield %44 : f32
      } -> tensor<1x?xf32>
      

      
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // 8. Define the rewrite callback
  transform.named_sequence @cast_and_call_dag(
    %ins: !transform.any_value {transform.readonly},
    %out: !transform.any_value {transform.readonly}
  ) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op

    // Import the executable and the replacement function
    %executable = transform.util.import_symbol @qgemm_executable into %module if undefined
      : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_qgemm_dequant into %module if undefined
      : (!transform.any_op) -> !transform.any_op

    // Cast inputs and call the replacement function
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // 9. Define the main transform entry point
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
    
    transform.foreach %funcs : !transform.any_op {
    ^bb1(%func: !transform.any_op):
      transform.foreach_match in %func
        @match_qgemm_bias_fused_scale -> @cast_and_call_dag
        : (!transform.any_op) -> (!transform.any_op)
    }
    
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}