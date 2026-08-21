// L9_gelu_f32: GELU (tanh approximation), f32 -- y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// Purely ELEMENTWISE: no reduction, no broadcast, no cross-element dependency of any kind. That is the
// whole point -- softmax bundles a hard transcendental together with a two-pass reduction, so failing
// it says nothing about which half is broken. `tanh` has no RISC-V instruction, so the backend must
// supply an approximation; the tolerance allows for that (vortex_oracle.F32_TANH_REL_ERR) and for
// nothing else.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "L9_gelu_f32"} {
  func.func @forward(%A: tensor<16x64xf32> {merlin.role = "input"}) -> tensor<16x64xf32> {
    %half = arith.constant 5.000000e-01 : f32
    %one = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 0.797884583 : f32
    %c2 = arith.constant 4.471500e-02 : f32
    %e = tensor.empty() : tensor<16x64xf32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A : tensor<16x64xf32>) outs(%e : tensor<16x64xf32>) {
    ^bb0(%a: f32, %o: f32):
      %x2 = arith.mulf %a, %a : f32
      %x3 = arith.mulf %x2, %a : f32
      %t0 = arith.mulf %x3, %c2 : f32
      %t1 = arith.addf %a, %t0 : f32
      %t2 = arith.mulf %t1, %c1 : f32
      %th = math.tanh %t2 : f32
      %t3 = arith.addf %th, %one : f32
      %t4 = arith.mulf %a, %t3 : f32
      %r = arith.mulf %t4, %half : f32
      linalg.yield %r : f32
    } -> tensor<16x64xf32>
    func.return %0 : tensor<16x64xf32>
  }
}
