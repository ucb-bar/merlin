// Matches `linalg.fill` with a constant-0 f32 input over a 3D output. The
// match captures the constant 0 inside the body via
// `match.operation_name_only` on tensor.empty so cast_compatible_dag_from_root
// only checks op identity for the empty (whose shape can vary across the
// payload).

^bb0():
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() {"match.operation_name_only"} : tensor<?x?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
