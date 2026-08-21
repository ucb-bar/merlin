// V15_divergent_nested_i8: nested data-dependent control flow -- an `scf.if` inside an unbounded `scf.while`.
// out[i,j] = sum of every value the Collatz trajectory visits, starting from n0 = (a[i,j] & 31) + 1.
// Two levels of divergence that must COMPOSE: lanes disagree on how many iterations to run (the outer
// while) AND, within an iteration, on which arm to take (the inner if). V10 has only the first -- its
// body is a branchless select -- so a backend can pass V10 with a single mask and still get this wrong
// by reconverging the inner region against the outer one. Integer-exact; the accumulator is the sum of
// visited values, so a wrong mask produces a wrong number rather than a coincidentally-right count.
//
// The 5-bit mask (n0 in 1..32) keeps trajectories short enough for cycle-exact simulation while still
// spreading trip counts across lanes.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V15_divergent_nested_i8"} {
  func.func @forward(%A: tensor<8x8xi8> {merlin.role = "input"}) -> tensor<8x8xi32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c31 = arith.constant 31 : i32
    %e = tensor.empty() : tensor<8x8xi32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A : tensor<8x8xi8>) outs(%e : tensor<8x8xi32>) {
    ^bb0(%a: i8, %o: i32):
      %ext = arith.extsi %a : i8 to i32
      %low = arith.andi %ext, %c31 : i32
      %n0 = arith.addi %low, %c1 : i32
      %res:2 = scf.while (%n = %n0, %acc = %c0) : (i32, i32) -> (i32, i32) {
        %cond = arith.cmpi ne, %n, %c1 : i32
        scf.condition(%cond) %n, %acc : i32, i32
      } do {
      ^bb1(%n1: i32, %s1: i32):
        %r = arith.remsi %n1, %c2 : i32
        %even = arith.cmpi eq, %r, %c0 : i32
        %next = scf.if %even -> (i32) {
          %h = arith.divsi %n1, %c2 : i32
          scf.yield %h : i32
        } else {
          %t = arith.muli %n1, %c3 : i32
          %t1 = arith.addi %t, %c1 : i32
          scf.yield %t1 : i32
        }
        %ss = arith.addi %s1, %next : i32
        scf.yield %next, %ss : i32, i32
      }
      linalg.yield %res#1 : i32
    } -> tensor<8x8xi32>
    func.return %0 : tensor<8x8xi32>
  }
}
