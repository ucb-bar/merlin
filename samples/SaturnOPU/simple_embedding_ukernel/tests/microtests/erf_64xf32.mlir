// Phase D6b: isolate math.erf (GELU activation).
// vit dispatches d11, d23 call math.erf. Usually lowered to a polynomial
// expansion on bare metal, but if the lowering emits a vfpoly intrinsic
// or a split-range branch that relies on vfredusum, it could hang.
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %init = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
       ins(%x : tensor<64xf32>) outs(%init : tensor<64xf32>) {
       ^bb0(%in: f32, %out: f32):
         %r = math.erf %in : f32
         linalg.yield %r : f32
       } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
