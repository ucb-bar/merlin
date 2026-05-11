// Phase D8: isolate f32 → i32 → f32 round-trip (vfcvt.x.f.v / vfcvt.f.x.v).
// d9's requant path does i32 matmul → f32 scale/bias → i8 out. If either
// conversion direction hangs, we see it here. Takes f32 input (matches
// the benchmark harness's default input type) and exercises both
// conversion directions inside the elementwise body.
func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %scale = arith.constant 0.125 : f32
  %init = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {
         indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
       ins(%x : tensor<64xf32>) outs(%init : tensor<64xf32>) {
       ^bb0(%in: f32, %out: f32):
         %i   = arith.fptosi %in : f32 to i32
         %f   = arith.sitofp %i : i32 to f32
         %fm  = arith.mulf %f, %scale : f32
         linalg.yield %fm : f32
       } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
