// L8_rmsnorm_f32: row-wise RMS normalisation, f32 -- y = x / sqrt(mean(x^2) + eps), eps = 1e-5.
// Softmax's structure with one difficulty removed: a single reduction along the row (not two of
// opposite kind), a rank-1 result broadcast back across rank 2, and a transcendental that IS a
// hardware instruction (`fsqrt`, correctly rounded) rather than one the backend must approximate.
// The reciprocal is NOT precomputed into a rank-1 tensor here -- it is folded into the broadcast
// consumer, so whether to hoist it out of the inner loop stays the compiler's decision.
#in2 = affine_map<(d0, d1) -> (d0, d1)>
#in1 = affine_map<(d0, d1) -> (d0)>
module attributes {merlin.capsule = "L8_rmsnorm_f32"} {
  func.func @forward(%A: tensor<16x64xf32> {merlin.role = "input"}) -> tensor<16x64xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %n = arith.constant 6.400000e+01 : f32
    %eps = arith.constant 9.99999975E-6 : f32
    %es = tensor.empty() : tensor<16xf32>
    %si = linalg.fill ins(%zero : f32) outs(%es : tensor<16xf32>) -> tensor<16xf32>
    %ss = linalg.generic {indexing_maps = [#in2, #in1],
                         iterator_types = ["parallel", "reduction"]}
         ins(%A : tensor<16x64xf32>) outs(%si : tensor<16xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %p = arith.mulf %a, %a : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
    } -> tensor<16xf32>
    %eo = tensor.empty() : tensor<16x64xf32>
    %0 = linalg.generic {indexing_maps = [#in2, #in1, #in2],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A, %ss : tensor<16x64xf32>, tensor<16xf32>) outs(%eo : tensor<16x64xf32>) {
    ^bb0(%a: f32, %s: f32, %o: f32):
      %m = arith.divf %s, %n : f32
      %me = arith.addf %m, %eps : f32
      %r = math.sqrt %me : f32
      %q = arith.divf %a, %r : f32
      linalg.yield %q : f32
    } -> tensor<16x64xf32>
    func.return %0 : tensor<16x64xf32>
  }
}
