// V4_reduce_rows_f32: row-wise sum reduction, f32. A rank-reducing reduction with NO matmul
// structure — the output is rank 1, so a backend that only knows how to tile a matmul has nothing to
// pattern-match. Reduction order is the compiler's choice; graded with a derived tolerance.
#red = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module attributes {merlin.capsule = "V4_reduce_rows_f32"} {
  func.func @forward(%A: tensor<8x16xf32> {merlin.role = "input"}) -> tensor<8xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<8xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<8xf32>) -> tensor<8xf32>
    %0 = linalg.generic {indexing_maps = [#red, #out],
                         iterator_types = ["parallel", "reduction"]}
         ins(%A : tensor<8x16xf32>) outs(%init : tensor<8xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %s = arith.addf %acc, %a : f32
      linalg.yield %s : f32
    } -> tensor<8xf32>
    func.return %0 : tensor<8xf32>
  }
}
