// Phase D6: isolate math.rsqrt → vfrsqrt7.v.
// vit LayerNorm dispatches (d1, d10, d13, d22) all go through this. If
// this binary hangs on Saturn FireSim, vfrsqrt7.v is the culprit and we
// must extend the scalarization gate in ConvertToLLVM.cpp.
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %cst_eps = arith.constant 1.0e-5 : f32
  %init = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
       ins(%x : tensor<64xf32>) outs(%init : tensor<64xf32>) {
       ^bb0(%in: f32, %out: f32):
         %s = arith.addf %in, %cst_eps : f32
         %r = math.rsqrt %s : f32
         linalg.yield %r : f32
       } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
