// Phase D-gap-4: full vit-d9 fusion pattern.
// Reproduces the 4-operand residual + bias + requantize chain:
//   i8_matmul(lhs, rhs) → i32
//   residual i8 → f32
//   (matmul_i32 * scale) + residual_f32 + bias_broadcast → i8
//
// CPULowerToUKernels emits iree_uk_opu_matmul for the contraction and
// leaves the residual+bias+requant as a separate vectorized generic.
// If THIS test hangs but the plain matmul_i8_64x128x128 passes, the
// bug is in the post-matmul fused generic (vfcvt.f.x.v, vfmul.vf,
// vfadd.vv, vfcvt.x.f.v chain).
func.func @main(%x: tensor<64x128xf32>) -> tensor<64x128xi8> {
  // LHS = fptosi(input) → i8
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
  // Constant RHS and residual + bias.
  %rhs = arith.constant dense<2> : tensor<128x128xi8>
  %residual = arith.constant dense<3> : tensor<64x128xi8>
  %bias = arith.constant dense<0.125> : tensor<128xf32>
  // Matmul i8×i8 → i32.
  %c0 = arith.constant 0 : i32
  %acc_init = tensor.empty() : tensor<64x128xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<64x128xi32>) -> tensor<64x128xi32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xi8>, tensor<128x128xi8>)
                      outs(%acc : tensor<64x128xi32>) -> tensor<64x128xi32>
  // Fused residual-add + bias-broadcast + requant to i8.
  %out_init = tensor.empty() : tensor<64x128xi8>
  %scale = arith.constant 0.0625 : f32
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mm, %residual, %bias : tensor<64x128xi32>, tensor<64x128xi8>, tensor<128xf32>)
    outs(%out_init : tensor<64x128xi8>) {
    ^bb0(%mv: i32, %rv: i8, %bv: f32, %ov: i8):
      %mf = arith.sitofp %mv : i32 to f32
      %ms = arith.mulf %mf, %scale : f32
      %rf = arith.sitofp %rv : i8 to f32
      %sum = arith.addf %ms, %rf : f32
      %fin = arith.addf %sum, %bv : f32
      %q = arith.fptosi %fin : f32 to i8
      linalg.yield %q : i8
    } -> tensor<64x128xi8>
  return %out : tensor<64x128xi8>
}
