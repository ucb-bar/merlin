// Linalg-DAG match pattern for fp32 ReLU over a 1×16 tensor.
//
// ReLU is `max(x, 0)`. IREE typically lowers this via `arith.maximumf`
// against a constant 0; we match that exact body shape.

// Match-scope `arith.constant` referenced from inside the linalg body.
// IREE patch in `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`
// makes the matcher walk operands captured into matched ops' regions
// from outer scope; before that patch, this pattern matched but tripped
// the IRMapping lookup during cast_and_call. Now matches the
// canonicalized form `linalg.generic { %s = arith.maximumf %a, %cst }`
// where %cst was hoisted out of the body by canonicalize.
^bb0(%arg0: tensor<1x16xf32>):
%cst = arith.constant 0.0 : f32
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
    %s = arith.maximumf %a, %cst : f32
    linalg.yield %s : f32
} -> tensor<1x16xf32>
