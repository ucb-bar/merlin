// L2_linear_requant_i8: linear (i8 x i8 -> i32) + requant.
// The epilogue is FUSED into the consumer here, not left as a separate pass over memory — whether the
// backend keeps it fused is its own decision, and one the cycle count will report on.
#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (k, n)>
#mC = affine_map<(m, n, k) -> (m, n)>
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "L2_linear_requant_i8"} {
  func.func @forward(%A: tensor<16x64xi8> {merlin.role = "input"},
                     %W: tensor<64x64xi8> {merlin.role = "weight"}) -> tensor<16x64xi8> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x64xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<16x64xi32>) -> tensor<16x64xi32>
    %mm = linalg.generic {indexing_maps = [#mA, #mB, #mC],
                         iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %W : tensor<16x64xi8>, tensor<64x64xi8>) outs(%init : tensor<16x64xi32>) {
    ^bb0(%a: i8, %b: i8, %acc_: i32):
      %ea = arith.extsi %a : i8 to i32
      %eb = arith.extsi %b : i8 to i32
      %p = arith.muli %ea, %eb : i32
      %s = arith.addi %acc_, %p : i32
      linalg.yield %s : i32
    } -> tensor<16x64xi32>
    %lo = arith.constant -128 : i32
    %hi = arith.constant 127 : i32
    %sh = arith.constant 9 : i32
    %eQ = tensor.empty() : tensor<16x64xi8>
    %q = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%mm : tensor<16x64xi32>) outs(%eQ : tensor<16x64xi8>) {
    ^bb0(%v: i32, %o: i8):
      %d = arith.shrsi %v, %sh : i32
      %c1 = arith.maxsi %d, %lo : i32
      %c2 = arith.minsi %c1, %hi : i32
      %t = arith.trunci %c2 : i32 to i8
      linalg.yield %t : i8
    } -> tensor<16x64xi8>
    func.return %q : tensor<16x64xi8>
  }
}
