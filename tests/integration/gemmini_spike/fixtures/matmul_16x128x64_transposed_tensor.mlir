// B-transposed probe fixture — matches dronet's actual matmul lowering form
// (linalg.matmul with explicit indexing_maps that put rhs in N×K layout).
//
// Standard linalg.matmul has B as K×N with implicit map
//   (d0,d1,d2) -> (d2,d1)
// IREE's preprocessing folds a transpose-of-B into the matmul by rewriting
// the rhs map to (d0,d1,d2) -> (d1,d2), which means B is laid out as N×K
// physically. The compiler must then emit CONFIG_EX with b_transpose=1
// so the Gemmini PE array indexes B correctly. Without bTranspose=1, the
// MVIN reads B with wrong row-stride and produces deterministic garbage —
// exactly the bit-stable wrong hash dronet × Gemmini was showing.
//
// With A=B=all-ones, K=64, expected per-cell value = 64.
func.func @matmul_16x128x64_transposed(%A: tensor<16x64xi8>, %B: tensor<128x64xi8>) -> tensor<16x128xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<16x128xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<16x128xi32>) -> tensor<16x128xi32>
  %res = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%A, %B : tensor<16x64xi8>, tensor<128x64xi8>)
      outs(%fill : tensor<16x128xi32>) -> tensor<16x128xi32>
  return %res : tensor<16x128xi32>
}
