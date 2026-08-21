// L5_linear_relu_f32: linear (f32) + relu. K=64 reduction, so grading is tolerance-based.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "L5_linear_relu_f32"} {
  func.func @forward(%A: tensor<16x64xf32> {merlin.role = "input"},
                     %W: tensor<64x64xf32> {merlin.role = "weight"}) -> tensor<16x64xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<16x64xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<16x64xf32>) -> tensor<16x64xf32>
    %mm = linalg.matmul ins(%A, %W : tensor<16x64xf32>, tensor<64x64xf32>) outs(%init : tensor<16x64xf32>) -> tensor<16x64xf32>
    %eR = tensor.empty() : tensor<16x64xf32>
    %relu = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%mm : tensor<16x64xf32>) outs(%eR : tensor<16x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %z : f32
      linalg.yield %r : f32
    } -> tensor<16x64xf32>
    func.return %relu : tensor<16x64xf32>
  }
}
