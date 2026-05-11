// Phase D9b: isolate vfdiv.vv (vector-vector floating-point divide).
// vit_small uses vfdiv.vv in softmax normalization / LayerNorm variance.
// large_mlp only uses vfdiv.vf (scalar denominator). Our other micro-tests
// happen to only exercise the scalar-denominator form, so this test plugs
// the coverage gap.
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  // Synthesize a per-element divisor by offset + small additive constant
  // so the compiler can't fold it into a scalar broadcast.
  %init_d = tensor.empty() : tensor<64xf32>
  %d = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
       ins(%x : tensor<64xf32>) outs(%init_d : tensor<64xf32>) {
       ^bb0(%in: f32, %out: f32):
         %k = arith.constant 1.125 : f32
         %o = arith.addf %in, %k : f32
         linalg.yield %o : f32
       } -> tensor<64xf32>
  %init = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
       ins(%x, %d : tensor<64xf32>, tensor<64xf32>) outs(%init : tensor<64xf32>) {
       ^bb0(%a: f32, %b: f32, %out: f32):
         %r = arith.divf %a, %b : f32
         linalg.yield %r : f32
       } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
