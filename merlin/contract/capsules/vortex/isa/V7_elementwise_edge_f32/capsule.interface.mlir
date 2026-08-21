// V7_elementwise_edge_f32: elementwise add, f32. No reduction, no divergence.
module attributes {merlin.capsule = "V7_elementwise_edge_f32"} {
  func.func @forward(%A: tensor<7x9xf32> {merlin.role = "input"},
                     %B: tensor<7x9xf32> {merlin.role = "input"}) -> tensor<7x9xf32> {
    %init = tensor.empty() : tensor<7x9xf32>
    %0 = linalg.add ins(%A, %B : tensor<7x9xf32>, tensor<7x9xf32>) outs(%init : tensor<7x9xf32>) -> tensor<7x9xf32>
    func.return %0 : tensor<7x9xf32>
  }
}
