// our_pipeline_async.mlir
// This pipeline module mimics pipeline_async.mlir, but calls into dronet and fastdepth modules.
// It is designed for asynchronous execution and expects input dimensions matching dronet.mlir and fastdepth.mlir.

// External function declarations for dronet and fastdepth modules.
// These must match the exported function names and signatures in the respective modules.
// The iree.abi.model attribute enables async support (coarse-fences ABI).
// FastDepth: expects input [1,1,224,224] and returns one output [1,1,224,224]

func.func private @fastdepth.main_graph$async(%input: tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xf32> attributes {
  iree.abi.model = "coarse-fences"
}
// Dronet: expects input [1,1,224,224] and returns two outputs [1,1], [1,1]
func.func private @dronet.model$async(%input: tensor<1x1x224x224xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>) attributes {
  iree.abi.model = "coarse-fences"
}

func.func private @simple_mlp.main_graph$async(%input: tensor<1x10xf32>) -> tensor<1x2xf32> attributes {
  iree.abi.model = "coarse-fences"
}

// Top-level pipeline function
// This function demonstrates calling both modules and combining their results.
func.func @run(
  %input: tensor<1x3x224x224xf32>,
  %mlp_input: tensor<1x10xf32>
) -> (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1x224x224xf32>, tensor<1x2xf32>) {
  // Call FastDepth for depth estimation
  %depth = call @fastdepth.main_graph$async(%input) : (tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xf32>

  // Call Dronet for steering and collision outputs
  %steer, %collision = call @dronet.model$async(%depth) : (tensor<1x1x224x224xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)

  // Call SimpleMLP independently
  %mlp_out = call @simple_mlp.main_graph$async(%mlp_input) : (tensor<1x10xf32>) -> tensor<1x2xf32>

  // Return all outputs as a tuple
  return %steer, %collision, %depth, %mlp_out : tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1x224x224xf32>, tensor<1x2xf32>
}

