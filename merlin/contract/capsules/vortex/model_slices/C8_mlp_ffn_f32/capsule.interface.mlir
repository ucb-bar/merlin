// C8_mlp_ffn_f32: transformer FFN, f32 -- Y = gelu(X @ W1) @ W2.
// The corpus's first TWO-MATMUL chain and the FFN half of a transformer block (H19 is attention). A
// backend that does linear (L4) and gelu (L9) alone must still compose them: the S x H intermediate has
// to stay live across the gelu into the second contraction. Graded with a derived composition tolerance.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "C8_mlp_ffn_f32"} {
  func.func @forward(%X: tensor<16x16xf32> {merlin.role = "input"},
                     %W1: tensor<16x32xf32> {merlin.role = "weight"},
                     %W2: tensor<32x16xf32> {merlin.role = "weight"}) -> tensor<16x16xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %half = arith.constant 5.000000e-01 : f32
    %one = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 0.797884583 : f32
    %c2 = arith.constant 4.471500e-02 : f32
    %eh = tensor.empty() : tensor<16x32xf32>
    %ih = linalg.fill ins(%z : f32) outs(%eh : tensor<16x32xf32>) -> tensor<16x32xf32>
    %hid = linalg.matmul ins(%X, %W1 : tensor<16x16xf32>, tensor<16x32xf32>) outs(%ih : tensor<16x32xf32>) -> tensor<16x32xf32>
    %acte = tensor.empty() : tensor<16x32xf32>
    %act = linalg.generic {indexing_maps = [#ew, #ew], iterator_types = ["parallel", "parallel"]}
         ins(%hid : tensor<16x32xf32>) outs(%acte : tensor<16x32xf32>) {
    ^bb0(%a: f32, %o: f32):
      %x2 = arith.mulf %a, %a : f32
      %x3 = arith.mulf %x2, %a : f32
      %g0 = arith.mulf %x3, %c2 : f32
      %g1 = arith.addf %a, %g0 : f32
      %g2 = arith.mulf %g1, %c1 : f32
      %gt = math.tanh %g2 : f32
      %g3 = arith.addf %gt, %one : f32
      %g4 = arith.mulf %a, %g3 : f32
      %gr = arith.mulf %g4, %half : f32
      linalg.yield %gr : f32
    } -> tensor<16x32xf32>
    %ey = tensor.empty() : tensor<16x16xf32>
    %iy = linalg.fill ins(%z : f32) outs(%ey : tensor<16x16xf32>) -> tensor<16x16xf32>
    %out = linalg.matmul ins(%act, %W2 : tensor<16x32xf32>, tensor<32x16xf32>) outs(%iy : tensor<16x16xf32>) -> tensor<16x16xf32>
    func.return %out : tensor<16x16xf32>
  }
}
