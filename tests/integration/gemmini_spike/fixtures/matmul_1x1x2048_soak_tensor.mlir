// Tensor-domain int8 matmul_1x1x2048 fixture.
//
// Same shape as dronet's linear1/linear2 FC heads (steer / collision).
// M=1, J=1, K=2048. Padded to dim=16 gives padI=15, padJ=15, padK=0;
// tileI=1, tileJ=1, tileK=128. tensor form so we go through IREE
// dispatch-creation + the gemmini lowering pipeline cleanly (the memref
// form crashes RaiseSpecialOpsPass with TypeRange[0] OOB).
//
// Use this to repro the 2026-05-17 8x steer-only numerical bug in
// isolation. With known constant A and B, the i32 output can be compared
// against `numpy.matmul(A_i8.astype(i32), B_i8.astype(i32))`.
//
// See: project_gemmini_steer_8x_bug.md
func.func @matmul_1x1x2048_soak(%A: tensor<1x2048xi8>, %B: tensor<2048x1xi8>) -> tensor<1x1xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<1x1xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<1x1xi32>) -> tensor<1x1xi32>
  %res = linalg.matmul ins(%A, %B : tensor<1x2048xi8>, tensor<2048x1xi8>)
                       outs(%fill : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %res : tensor<1x1xi32>
}
