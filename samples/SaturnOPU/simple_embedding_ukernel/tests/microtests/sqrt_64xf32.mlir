// Phase D7: isolate math.sqrt → vfsqrt.v.
// Sanity companion to rsqrt; if rsqrt hangs but sqrt passes, vfrsqrt7
// is broken but vfsqrt works.
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
         %r = math.sqrt %s : f32
         linalg.yield %r : f32
       } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
