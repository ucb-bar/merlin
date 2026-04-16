// Phase D-gap-5: matmul with accumulate=true (per-row VMV_RV reload).
// Exercises the accumulate path in iree_uk_opu_matmul lines 238-263
// that per-row loads existing output via vle32.v + VMV_RV m0/m2.
// Normal fill-0 init goes accumulate=false and skips this path entirely.
// We construct a non-zero init by casting the f32 input to i32 and
// using it as the initial accumulator.
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
  // Non-zero initial accumulator from input — forces accumulate=true.
  %acc_init = tensor.empty() : tensor<64x128xi32>
  %acc = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<64x128xf32>) outs(%acc_init : tensor<64x128xi32>) {
    ^bb0(%in: f32, %out: i32):
      %i = arith.fptosi %in : f32 to i32
      linalg.yield %i : i32
    } -> tensor<64x128xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x128xi8>, tensor<128x128xi8>)
                     outs(%acc : tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}
