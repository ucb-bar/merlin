func.func private @run_vision(%input: tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xf32> attributes { iree.abi.model = "coarse-fences" }
func.func private @run_mlp(%input: tensor<1x10xf32>) -> tensor<1x2xf32> attributes { iree.abi.model = "coarse-fences" }

func.func @run_looped(
  %vision_input: tensor<1x3x224x224xf32>,
  %mlp_input: tensor<1x10xf32>
) -> (tensor<1x1x224x224xf32>, tensor<1x2xf32>) {
  // Call vision module
  %vision_result = call @run_vision(%vision_input) : (tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xf32>

  // Loop: call MLP N times (e.g., 10)
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %mlp_result_init = call @run_mlp(%mlp_input) : (tensor<1x10xf32>) -> tensor<1x2xf32>
  %final_mlp = scf.for %i = %c0 to %c10 step %c1 iter_args(%mlp_acc = %mlp_result_init) -> (tensor<1x2xf32>) {
    %mlp_res = call @run_mlp(%mlp_input) : (tensor<1x10xf32>) -> tensor<1x2xf32>
    scf.yield %mlp_res : tensor<1x2xf32>
  }

  return %vision_result, %final_mlp : tensor<1x1x224x224xf32>, tensor<1x2xf32>
}
