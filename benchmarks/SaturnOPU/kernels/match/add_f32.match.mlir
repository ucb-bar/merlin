// Matches a 1-D linalg.generic f32 elementwise add with dynamic shape. The
// auto-generated cast_and_call sequence inserts tensor.cast to bridge to
// statically-shaped payload (see
// transform.type_conversion.tensor.cast_shape_dynamic_dims).

^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
  %add = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
