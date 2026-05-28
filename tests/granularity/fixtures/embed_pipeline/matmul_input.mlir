// Synthetic input that matches `saturnopu_linear_f32`: a 2D f32 matmul
// expressed as a linalg.generic with transposed-B indexing maps
//   out[m, n] = sum_k lhs[m, k] * rhs[n, k]
// (i.e. lhs ∈ MxK row-major, rhs ∈ NxK row-major).
//
// Used by:
//   * benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh
//   * tests/granularity/test_phase_dumps_fresh.py

util.func public @main(
    %lhs: tensor<32x32xf32>,
    %rhs: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %init = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %r = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
    ^bb_inner(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
  } -> tensor<32x32xf32>
  util.return %r : tensor<32x32xf32>
}
