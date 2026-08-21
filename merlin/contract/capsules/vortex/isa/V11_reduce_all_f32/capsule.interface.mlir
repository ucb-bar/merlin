// V11_reduce_all_f32: whole-tensor sum reduction (16x16 -> a single scalar).
// BOTH iterators are "reduction", so there is exactly ONE output element for 256 inputs. Every
// other reduction in the corpus gives each thread its own output row and needs no communication; this
// one does not, so a multi-thread mapping has to combine partial results ACROSS lanes -- via shuffles,
// via the scratchpad with a barrier, or via an atomic. A single-coordinate scalar loop is still a
// LEGAL answer (nothing in an input can forbid it) and will pass; what it will not do is be fast, and
// the cycle count is where that shows.
#in2 = affine_map<(d0, d1) -> (d0, d1)>
#out0 = affine_map<(d0, d1) -> (0)>
module attributes {merlin.capsule = "V11_reduce_all_f32"} {
  func.func @forward(%A: tensor<16x16xf32> {merlin.role = "input"}) -> tensor<1xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<1xf32>) -> tensor<1xf32>
    %0 = linalg.generic {indexing_maps = [#in2, #out0],
                         iterator_types = ["reduction", "reduction"]}
         ins(%A : tensor<16x16xf32>) outs(%init : tensor<1xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %s = arith.addf %acc, %a : f32
      linalg.yield %s : f32
    } -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
}
