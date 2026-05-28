// Matches the standard `linalg.matmul` named op over f32 with untransposed
// RHS — out[m, n] = sum_k lhs[m, k] * rhs[k, n]. This is the form dronet's
// final classifier dot ends up in after im2col preprocessing (and with
// data tiling disabled so the encoding annotations don't appear).
//
// Body uses `linalg.matmul` directly (not the generic form) because IREE's
// preprocessing keeps it as a named op when standard layout matches.

^bb0(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %n = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) {"match.operation_name_only"} : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
