// Linalg-DAG match pattern for fp32 elementwise sigmoid.
//
// QNN's `Sigmoid` op consumes one fp32 input, produces one same-shape
// fp32 output. We match the canonical IREE form: a `linalg.generic` with
// a scalar body computing `1 / (1 + exp(-x))` (or the equivalent
// `exp(x) / (1+exp(x))` form depending on the upstream lowering).
//
// Both forms compile to the same arith chain inside the body. We pick
// the more numerically stable one as the literal pattern; if upstream
// emits the other form, this manifest entry needs a sibling pattern.

// Match-scope constant captured into the linalg body — works after the
// IREE matcher patch in PreprocessingExtensions.cpp (#114).
^bb0(%arg0: tensor<1x16xf32>):
%cst_one = arith.constant 1.0 : f32
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
    // Operand order matches IREE's canonical lowering of sigmoid:
    // `addf %exp, %cst` then `divf %cst, %denom`. The matcher does a
    // strict operand-order compare, so flipping these would mis-match.
    %neg = arith.negf %a : f32
    %exp = math.exp %neg : f32
    %denom = arith.addf %exp, %cst_one : f32
    %s = arith.divf %cst_one, %denom : f32
    linalg.yield %s : f32
} -> tensor<1x16xf32>
