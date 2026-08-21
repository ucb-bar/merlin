// C9_attention_sublayer_f32: attention SUBLAYER, f32 -- Y = rmsnorm(softmax(Q @ K^T) @ V + Q).
// The heaviest composition: the H19 attention block, then a RESIDUAL add of the input, then a final RMS
// norm -- five stages of different character (transposed contraction, two-reduction softmax,
// contraction, elementwise residual, reduction-normalise). L2 ONLY: even a correct naive SIMT mapping of
// the attention here runs into the millions of cycles (see H19's IPC=0.069), past the rtlsim wall.
#qA = affine_map<(i, j, p) -> (i, p)>
#qK = affine_map<(i, j, p) -> (j, p)>
#qS = affine_map<(i, j, p) -> (i, j)>
#s2 = affine_map<(d0, d1) -> (d0, d1)>
#s1 = affine_map<(d0, d1) -> (d0)>
module attributes {merlin.capsule = "C9_attention_sublayer_f32"} {
  func.func @forward(%Q: tensor<12x16xf32> {merlin.role = "input"},
                     %K: tensor<12x16xf32> {merlin.role = "input"},
                     %V: tensor<12x16xf32> {merlin.role = "input"}) -> tensor<12x16xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %ninf = arith.constant -3.40282347E+38 : f32
    %nrm = arith.constant 1.600000e+01 : f32
    %eps = arith.constant 9.99999975E-6 : f32
    %esc = tensor.empty() : tensor<12x12xf32>
    %sc0 = linalg.fill ins(%zero : f32) outs(%esc : tensor<12x12xf32>) -> tensor<12x12xf32>
    %scores = linalg.generic {indexing_maps = [#qA, #qK, #qS], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%Q, %K : tensor<12x16xf32>, tensor<12x16xf32>) outs(%sc0 : tensor<12x12xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
      %p = arith.mulf %q, %k : f32
      %a = arith.addf %acc, %p : f32
      linalg.yield %a : f32
    } -> tensor<12x12xf32>
    %em = tensor.empty() : tensor<12xf32>
    %mi = linalg.fill ins(%ninf : f32) outs(%em : tensor<12xf32>) -> tensor<12xf32>
    %mx = linalg.generic {indexing_maps = [#s2, #s1], iterator_types = ["parallel", "reduction"]}
         ins(%scores : tensor<12x12xf32>) outs(%mi : tensor<12xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %m = arith.maximumf %acc, %v : f32
      linalg.yield %m : f32
    } -> tensor<12xf32>
    %ee = tensor.empty() : tensor<12x12xf32>
    %ex = linalg.generic {indexing_maps = [#s2, #s1, #s2], iterator_types = ["parallel", "parallel"]}
         ins(%scores, %mx : tensor<12x12xf32>, tensor<12xf32>) outs(%ee : tensor<12x12xf32>) {
    ^bb0(%v: f32, %m: f32, %o: f32):
      %sub = arith.subf %v, %m : f32
      %e0 = math.exp %sub : f32
      linalg.yield %e0 : f32
    } -> tensor<12x12xf32>
    %es = tensor.empty() : tensor<12xf32>
    %si = linalg.fill ins(%zero : f32) outs(%es : tensor<12xf32>) -> tensor<12xf32>
    %sm = linalg.generic {indexing_maps = [#s2, #s1], iterator_types = ["parallel", "reduction"]}
         ins(%ex : tensor<12x12xf32>) outs(%si : tensor<12xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %a = arith.addf %acc, %v : f32
      linalg.yield %a : f32
    } -> tensor<12xf32>
    %ep = tensor.empty() : tensor<12x12xf32>
    %probs = linalg.generic {indexing_maps = [#s2, #s1, #s2], iterator_types = ["parallel", "parallel"]}
         ins(%ex, %sm : tensor<12x12xf32>, tensor<12xf32>) outs(%ep : tensor<12x12xf32>) {
    ^bb0(%v: f32, %su: f32, %o: f32):
      %qd = arith.divf %v, %su : f32
      linalg.yield %qd : f32
    } -> tensor<12x12xf32>
    %eat = tensor.empty() : tensor<12x16xf32>
    %iat = linalg.fill ins(%zero : f32) outs(%eat : tensor<12x16xf32>) -> tensor<12x16xf32>
    %attn = linalg.matmul ins(%probs, %V : tensor<12x12xf32>, tensor<12x16xf32>) outs(%iat : tensor<12x16xf32>) -> tensor<12x16xf32>
    %ers = tensor.empty() : tensor<12x16xf32>
    %res = linalg.generic {indexing_maps = [#s2, #s2, #s2], iterator_types = ["parallel", "parallel"]}
         ins(%attn, %Q : tensor<12x16xf32>, tensor<12x16xf32>) outs(%ers : tensor<12x16xf32>) {
    ^bb0(%aa: f32, %qq: f32, %o: f32):
      %sm0 = arith.addf %aa, %qq : f32
      linalg.yield %sm0 : f32
    } -> tensor<12x16xf32>
    %ers2 = tensor.empty() : tensor<12xf32>
    %rsi = linalg.fill ins(%zero : f32) outs(%ers2 : tensor<12xf32>) -> tensor<12xf32>
    %rss = linalg.generic {indexing_maps = [#s2, #s1], iterator_types = ["parallel", "reduction"]}
         ins(%res : tensor<12x16xf32>) outs(%rsi : tensor<12xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %p = arith.mulf %a, %a : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
    } -> tensor<12xf32>
    %eo = tensor.empty() : tensor<12x16xf32>
    %out = linalg.generic {indexing_maps = [#s2, #s1, #s2], iterator_types = ["parallel", "parallel"]}
         ins(%res, %rss : tensor<12x16xf32>, tensor<12xf32>) outs(%eo : tensor<12x16xf32>) {
    ^bb0(%a: f32, %s: f32, %o: f32):
      %m = arith.divf %s, %nrm : f32
      %me = arith.addf %m, %eps : f32
      %r = math.sqrt %me : f32
      %q = arith.divf %a, %r : f32
      linalg.yield %q : f32
    } -> tensor<12x16xf32>
    func.return %out : tensor<12x16xf32>
  }
}
