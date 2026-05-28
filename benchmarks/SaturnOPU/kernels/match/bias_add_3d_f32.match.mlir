// Matches dronet's per-conv bias-add pattern:
//   out[c, h, w] = in[c, h, w] + bias[c]
// Indexing maps:
//   #map3 = (d0, d1, d2) -> (d0, d1, d2)
//   #map4 = (d0, d1, d2) -> (d0)

^bb0(%in: tensor<?x?x?xf32>, %bias: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %in, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %in, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %in, %c2 : tensor<?x?x?xf32>
  %empty = tensor.empty(%dim0, %dim1, %dim2) {"match.operation_name_only"} : tensor<?x?x?xf32>
  %add = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%in, %bias : tensor<?x?x?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?x?x?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?x?x?xf32>
