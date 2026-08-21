// V5_reduce_rows_i8: row-wise sum reduction, i8 widened to i32. Exact — cols=16 cannot overflow.
#red = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module attributes {merlin.capsule = "V5_reduce_rows_i8"} {
  func.func @forward(%A: tensor<8x16xi8> {merlin.role = "input"}) -> tensor<8xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
    %0 = linalg.generic {indexing_maps = [#red, #out],
                         iterator_types = ["parallel", "reduction"]}
         ins(%A : tensor<8x16xi8>) outs(%init : tensor<8xi32>) {
    ^bb0(%a: i8, %acc: i32):
      %ea = arith.extsi %a : i8 to i32
      %s = arith.addi %acc, %ea : i32
      linalg.yield %s : i32
    } -> tensor<8xi32>
    func.return %0 : tensor<8xi32>
  }
}
