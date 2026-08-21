// V1_elementwise_add_i8: elementwise add, i8 operands widened to an i32 result. Exact integer.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V1_elementwise_add_i8"} {
  func.func @forward(%A: tensor<8x8xi8> {merlin.role = "input"},
                     %B: tensor<8x8xi8> {merlin.role = "input"}) -> tensor<8x8xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<8x8xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<8x8xi32>) -> tensor<8x8xi32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A, %B : tensor<8x8xi8>, tensor<8x8xi8>) outs(%init : tensor<8x8xi32>) {
    ^bb0(%a: i8, %b: i8, %o: i32):
      %ea = arith.extsi %a : i8 to i32
      %eb = arith.extsi %b : i8 to i32
      %s = arith.addi %ea, %eb : i32
      linalg.yield %s : i32
    } -> tensor<8x8xi32>
    func.return %0 : tensor<8x8xi32>
  }
}
