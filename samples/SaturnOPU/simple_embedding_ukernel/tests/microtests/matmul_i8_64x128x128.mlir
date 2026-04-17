// Phase D-gap-1: i8×i8→i32 matmul at vit d9 shape (64×128×128).
// This is the dtype path the real vit and tinyllama dispatches take.
// f32 input keeps the harness happy; we cast to i8 inside so the
// constant RHS isn't folded at compile time.
func.func @main(%x: tensor<64x128xf32>) -> tensor<64x128xi32> {
  %lhs_init = tensor.empty() : tensor<64x128xi8>
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<64x128xf32>) outs(%lhs_init : tensor<64x128xi8>) {
    ^bb0(%in: f32, %out: i8):
      %i = arith.fptosi %in : f32 to i8
      linalg.yield %i : i8
    } -> tensor<64x128xi8>
  %rhs = arith.constant dense<2> : tensor<128x128xi8>
  %c0 = arith.constant 0 : i32
  %acc_init = tensor.empty() : tensor<64x128xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<64x128xi32>) -> tensor<64x128xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x128xi8>, tensor<128x128xi8>)
                     outs(%acc : tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}
