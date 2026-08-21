// V6_matmul_transposed_i8: matmul against a TRANSPOSED weight, i8 -> i32. The weight is stored (n, k) and
// read as W[j, p]: the contraction walks it along its FAST axis, so a backend that assumed a (k, n)
// layout produces transposed garbage. No transpose op is present to pattern-match on.
#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (n, k)>
#mC = affine_map<(m, n, k) -> (m, n)>
module attributes {merlin.capsule = "V6_matmul_transposed_i8"} {
  func.func @forward(%A: tensor<8x16xi8> {merlin.role = "input"},
                     %W: tensor<8x16xi8> {merlin.role = "weight"}) -> tensor<8x8xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8x8xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<8x8xi32>) -> tensor<8x8xi32>
    %0 = linalg.generic {indexing_maps = [#mA, #mB, #mC],
                         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %W : tensor<8x16xi8>, tensor<8x16xi8>) outs(%init : tensor<8x8xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ea = arith.extsi %a : i8 to i32
      %eb = arith.extsi %b : i8 to i32
      %p = arith.muli %ea, %eb : i32
      %s = arith.addi %acc, %p : i32
      linalg.yield %s : i32
    } -> tensor<8x8xi32>
    func.return %0 : tensor<8x8xi32>
  }
}
