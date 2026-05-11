// Phase D-gap-2: narrow-M i8 matmul (M=1, ukernel narrow path).
// Exercises iree_uk_opu_matmul lines 322-368 (NARROW PATH at
// `m_hw0 != HW || n_hw0 != HW`). vit dispatches with batch-attention
// reshape end up here when the encoding resolver picks M0=1 tiles.
func.func @main(%x: tensor<1x128xf32>) -> tensor<1x128xi32> {
  %lhs_init = tensor.empty() : tensor<1x128xi8>
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<1x128xf32>) outs(%lhs_init : tensor<1x128xi8>) {
    ^bb0(%in: f32, %out: i8):
      %i = arith.fptosi %in : f32 to i8
      linalg.yield %i : i8
    } -> tensor<1x128xi8>
  %rhs = arith.constant dense<2> : tensor<128x128xi8>
  %c0 = arith.constant 0 : i32
  %acc_init = tensor.empty() : tensor<1x128xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<1x128xi32>) -> tensor<1x128xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<1x128xi8>, tensor<128x128xi8>)
                     outs(%acc : tensor<1x128xi32>) -> tensor<1x128xi32>
  return %0 : tensor<1x128xi32>
}
