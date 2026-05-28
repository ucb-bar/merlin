// Linalg-DAG match pattern for fp32 ReLU6 over a 1×16 tensor.
// ReLU6 is `min(max(x, 0), 6)`. Common in MobileNet-family activations.

// Match-scope constants captured into the linalg body (#114).
^bb0(%arg0: tensor<1x16xf32>):
%cst_zero = arith.constant 0.0 : f32
%cst_six = arith.constant 6.0 : f32
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x16xf32>
%out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<1x16xf32>)
    outs(%init : tensor<1x16xf32>) {
  ^bb0(%a: f32, %o: f32):
    %lo = arith.maximumf %a, %cst_zero : f32
    %s = arith.minimumf %lo, %cst_six : f32
    linalg.yield %s : f32
} -> tensor<1x16xf32>
