// Linalg-DAG match pattern for fp32 elementwise multiply over a 1×16 tensor.

^bb0(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>):
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x16xf32>
%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<1x16xf32>, tensor<1x16xf32>)
    outs(%init : tensor<1x16xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.mulf %a, %b : f32
    linalg.yield %s : f32
} -> tensor<1x16xf32>
