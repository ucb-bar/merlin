// V3_matmul_f32: matmul, f32. A K=16 reduction — reduction ORDER is the compiler's choice,
// which is why this capsule grades with a derived tolerance rather than bit-exactly.
module attributes {merlin.capsule = "V3_matmul_f32"} {
  func.func @forward(%A: tensor<8x16xf32> {merlin.role = "input"},
                     %W: tensor<16x8xf32> {merlin.role = "weight"}) -> tensor<8x8xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<8x8xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
    %0 = linalg.matmul ins(%A, %W : tensor<8x16xf32>, tensor<16x8xf32>) outs(%init : tensor<8x8xf32>) -> tensor<8x8xf32>
    func.return %0 : tensor<8x8xf32>
  }
}
