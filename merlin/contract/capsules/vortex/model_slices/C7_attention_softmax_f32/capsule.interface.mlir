// C7_attention_softmax_f32: numerically-stable row-wise softmax, f32 — max-subtract, exp, sum, divide.
// THE capsule a systolic tensor engine cannot express and a SIMT core can. Three things make it the
// hardest input in the corpus: two reductions of OPPOSITE kind (max then sum) over the same axis, a
// rank-1 intermediate broadcast back across a rank-2 tensor, and a transcendental with no hardware
// instruction. Grading allows for the backend's own exp approximation; see F32_EXP_REL_ERR.
#in2 = affine_map<(d0, d1) -> (d0, d1)>
#in1 = affine_map<(d0, d1) -> (d0)>
module attributes {merlin.capsule = "C7_attention_softmax_f32"} {
  func.func @forward(%A: tensor<16x16xf32> {merlin.role = "input"}) -> tensor<16x16xf32> {
    %ninf = arith.constant -3.40282347E+38 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %em = tensor.empty() : tensor<16xf32>
    %mi = linalg.fill ins(%ninf : f32) outs(%em : tensor<16xf32>) -> tensor<16xf32>
    %mx = linalg.generic {indexing_maps = [#in2, #in1],
                         iterator_types = ["parallel", "reduction"]}
         ins(%A : tensor<16x16xf32>) outs(%mi : tensor<16xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.maximumf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<16xf32>
    %ee = tensor.empty() : tensor<16x16xf32>
    %ex = linalg.generic {indexing_maps = [#in2, #in1, #in2],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A, %mx : tensor<16x16xf32>, tensor<16xf32>) outs(%ee : tensor<16x16xf32>) {
    ^bb0(%a: f32, %m: f32, %o: f32):
      %d = arith.subf %a, %m : f32
      %x = math.exp %d : f32
      linalg.yield %x : f32
    } -> tensor<16x16xf32>
    %es = tensor.empty() : tensor<16xf32>
    %si = linalg.fill ins(%zero : f32) outs(%es : tensor<16xf32>) -> tensor<16xf32>
    %sm = linalg.generic {indexing_maps = [#in2, #in1],
                         iterator_types = ["parallel", "reduction"]}
         ins(%ex : tensor<16x16xf32>) outs(%si : tensor<16xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %s = arith.addf %acc, %v : f32
      linalg.yield %s : f32
    } -> tensor<16xf32>
    %eo = tensor.empty() : tensor<16x16xf32>
    %0 = linalg.generic {indexing_maps = [#in2, #in1, #in2],
                         iterator_types = ["parallel", "parallel"]}
         ins(%ex, %sm : tensor<16x16xf32>, tensor<16xf32>) outs(%eo : tensor<16x16xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %q = arith.divf %v, %s : f32
      linalg.yield %q : f32
    } -> tensor<16x16xf32>
    func.return %0 : tensor<16x16xf32>
  }
}
