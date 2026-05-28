// Matches a 2D f32 matmul with transposed B (rhs interpreted as (N, K)
// row-major). Expressed as a linalg.generic with explicit indexing maps
// rather than `linalg.matmul_transpose_b` because not every in-tree IREE
// version registers that named op. Equivalent semantics:
//   out[m, n] = sum_k lhs[m, k] * rhs[n, k]
// Matches the convention `kernel_linear` uses in `../rvv_linear_direct.c`:
// `weight + n * K` to scan a column of the conceptual matmul.

^bb0(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>):
  %c0 = arith.constant 0 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) {"match.operation_name_only"} : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%filled : tensor<?x?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
  } -> tensor<?x?xf32>
