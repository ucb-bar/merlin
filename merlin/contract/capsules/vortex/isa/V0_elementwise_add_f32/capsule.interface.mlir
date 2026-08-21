// V0_elementwise_add_f32: elementwise add, f32. No reduction, no divergence.
module attributes {merlin.capsule = "V0_elementwise_add_f32"} {
  func.func @forward(%A: tensor<8x8xf32> {merlin.role = "input"},
                     %B: tensor<8x8xf32> {merlin.role = "input"}) -> tensor<8x8xf32> {
    %init = tensor.empty() : tensor<8x8xf32>
    %0 = linalg.add ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    func.return %0 : tensor<8x8xf32>
  }
}
