// V8_matmul_edge_i8: matmul, i8 operands accumulated exactly in i32 (K=13; no overflow possible).
#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (k, n)>
#mC = affine_map<(m, n, k) -> (m, n)>
module attributes {merlin.capsule = "V8_matmul_edge_i8"} {
  func.func @forward(%A: tensor<5x13xi8> {merlin.role = "input"},
                     %W: tensor<13x7xi8> {merlin.role = "weight"}) -> tensor<5x7xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<5x7xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<5x7xi32>) -> tensor<5x7xi32>
    %0 = linalg.generic {indexing_maps = [#mA, #mB, #mC],
                         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %W : tensor<5x13xi8>, tensor<13x7xi8>) outs(%init : tensor<5x7xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ea = arith.extsi %a : i8 to i32
      %eb = arith.extsi %b : i8 to i32
      %p = arith.muli %ea, %eb : i32
      %s = arith.addi %acc, %p : i32
      linalg.yield %s : i32
    } -> tensor<5x7xi32>
    func.return %0 : tensor<5x7xi32>
  }
}
