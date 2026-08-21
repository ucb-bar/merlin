// V13_transpose_i8: transpose, i8 -> i32 (32x16 -> 16x32).
// Y[j, i] = A[i, j]. The arithmetic is a single sign-extend: this capsule is about ADDRESSING, not
// compute. Whichever axis the backend maps to the fast-varying thread index, the other side of the
// copy is strided, so the cycle count separates a staged/tiled transpose from a naive one by a wide
// margin. Read the source through a transposed access map rather than a `linalg.transpose` op, so
// there is no named op to pattern-match and special-case.
#src = affine_map<(d0, d1) -> (d1, d0)>
#dst = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V13_transpose_i8"} {
  func.func @forward(%A: tensor<32x16xi8> {merlin.role = "input"}) -> tensor<16x32xi32> {
    %e = tensor.empty() : tensor<16x32xi32>
    %0 = linalg.generic {indexing_maps = [#src, #dst],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A : tensor<32x16xi8>) outs(%e : tensor<16x32xi32>) {
    ^bb0(%a: i8, %o: i32):
      %ea = arith.extsi %a : i8 to i32
      linalg.yield %ea : i32
    } -> tensor<16x32xi32>
    func.return %0 : tensor<16x32xi32>
  }
}
