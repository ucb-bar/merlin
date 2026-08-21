// V10_divergent_iterate_i8: per-element Collatz step count -- an unbounded data-dependent loop.
// out[i,j] = number of steps to reach 1 from n0 = (a[i,j] & 127) + 1, where each step is
// n <- n/2 if n even else 3n+1. Trip count varies per element and cannot be bounded statically, so
// this is the capsule that genuinely forces divergence handling (split/join, or predication) rather
// than admitting a branchless rewrite. Integer-exact, so it grades bit-exactly.
//
// The low-7-bit mask (rather than a+1 clamped at 1) is deliberate: it maps every input to a distinct
// starting point in 1..128, so almost no lane exits at zero iterations. Clamping instead sent every
// negative input to n0=1, which left ~45% of elements doing no work at all.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V10_divergent_iterate_i8"} {
  func.func @forward(%A: tensor<8x8xi8> {merlin.role = "input"}) -> tensor<8x8xi32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c127 = arith.constant 127 : i32
    %e = tensor.empty() : tensor<8x8xi32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A : tensor<8x8xi8>) outs(%e : tensor<8x8xi32>) {
    ^bb0(%a: i8, %o: i32):
      %ext = arith.extsi %a : i8 to i32
      %low = arith.andi %ext, %c127 : i32
      %n0 = arith.addi %low, %c1 : i32
      %res:2 = scf.while (%n = %n0, %steps = %c0) : (i32, i32) -> (i32, i32) {
        %cond = arith.cmpi ne, %n, %c1 : i32
        scf.condition(%cond) %n, %steps : i32, i32
      } do {
      ^bb1(%n1: i32, %s1: i32):
        %r = arith.remsi %n1, %c2 : i32
        %even = arith.cmpi eq, %r, %c0 : i32
        %half = arith.divsi %n1, %c2 : i32
        %tri = arith.muli %n1, %c3 : i32
        %tri1 = arith.addi %tri, %c1 : i32
        %nn = arith.select %even, %half, %tri1 : i32
        %ss = arith.addi %s1, %c1 : i32
        scf.yield %nn, %ss : i32, i32
      }
      linalg.yield %res#1 : i32
    } -> tensor<8x8xi32>
    func.return %0 : tensor<8x8xi32>
  }
}
