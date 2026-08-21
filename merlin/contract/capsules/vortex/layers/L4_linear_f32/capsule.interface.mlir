// L4_linear_f32: linear (f32). K=64 reduction, so grading is tolerance-based.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "L4_linear_f32"} {
  func.func @forward(%A: tensor<16x64xf32> {merlin.role = "input"},
                     %W: tensor<64x64xf32> {merlin.role = "weight"}) -> tensor<16x64xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<16x64xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<16x64xf32>) -> tensor<16x64xf32>
    %mm = linalg.matmul ins(%A, %W : tensor<16x64xf32>, tensor<64x64xf32>) outs(%init : tensor<16x64xf32>) -> tensor<16x64xf32>
    func.return %mm : tensor<16x64xf32>
  }
}
