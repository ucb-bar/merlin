// Phase D-gap-3: wide i8 matmul (128×256×256).
// Exercises the sub32x32_full fast path in iree_uk_opu_matmul
// (`m_rem >= 2*HW && n_rem >= 2*HW && K0 == 1`). Our square 64×128
// test stays in the generic-fast branch; this one triggers the
// specialized 32×32 quadrant unroll (lines 53-171 of opu_matmul_riscv_64.c).
func.func @main(%x: tensor<128x256xf32>) -> tensor<128x256xi32> {
  %lhs_init = tensor.empty() : tensor<128x256xi8>
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x : tensor<128x256xf32>) outs(%lhs_init : tensor<128x256xi8>) {
    ^bb0(%in: f32, %out: i8):
      %i = arith.fptosi %in : f32 to i8
      linalg.yield %i : i8
    } -> tensor<128x256xi8>
  %rhs = arith.constant dense<2> : tensor<256x256xi8>
  %c0 = arith.constant 0 : i32
  %acc_init = tensor.empty() : tensor<128x256xi32>
  %acc = linalg.fill ins(%c0 : i32) outs(%acc_init : tensor<128x256xi32>) -> tensor<128x256xi32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<128x256xi8>, tensor<256x256xi8>)
                     outs(%acc : tensor<128x256xi32>) -> tensor<128x256xi32>
  return %0 : tensor<128x256xi32>
}
