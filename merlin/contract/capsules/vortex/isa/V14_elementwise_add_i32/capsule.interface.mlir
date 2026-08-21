// V14_elementwise_add_i32: elementwise add, i32 operands and an i32 result. Exact.
// The ONLY capsule whose INPUTS are 4 bytes wide: a load path hardcoded to `lb`+`extsi` (which every
// i8 capsule in the corpus rewards) reads the wrong bytes here. No widening is involved at all.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V14_elementwise_add_i32"} {
  func.func @forward(%A: tensor<8x8xi32> {merlin.role = "input"},
                     %B: tensor<8x8xi32> {merlin.role = "input"}) -> tensor<8x8xi32> {
    %e = tensor.empty() : tensor<8x8xi32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A, %B : tensor<8x8xi32>, tensor<8x8xi32>) outs(%e : tensor<8x8xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<8x8xi32>
    func.return %0 : tensor<8x8xi32>
  }
}
