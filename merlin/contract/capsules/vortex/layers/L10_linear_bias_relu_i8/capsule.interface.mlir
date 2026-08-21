// L10_linear_bias_relu_i8: linear (i8 x i8 -> i32) + bias + relu.
// The epilogue is FUSED into the consumer here, not left as a separate pass over memory — whether the
// backend keeps it fused is its own decision, and one the cycle count will report on.
#mA = affine_map<(m, n, k) -> (m, k)>
#mB = affine_map<(m, n, k) -> (k, n)>
#mC = affine_map<(m, n, k) -> (m, n)>
#ew = affine_map<(d0, d1) -> (d0, d1)>
#bc = affine_map<(d0, d1) -> (d1)>
module attributes {merlin.capsule = "L10_linear_bias_relu_i8"} {
  func.func @forward(%A: tensor<16x64xi8> {merlin.role = "input"},
                     %W: tensor<64x64xi8> {merlin.role = "weight"},
                     %BIAS: tensor<64xi32> {merlin.role = "bias"}) -> tensor<16x64xi32> {
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
    %eB = tensor.empty() : tensor<16x64xi32>
    %bias = linalg.generic {indexing_maps = [#ew, #bc, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%mm, %BIAS : tensor<16x64xi32>, tensor<64xi32>) outs(%eB : tensor<16x64xi32>) {
    ^bb0(%v: i32, %b_: i32, %o: i32):
      %sum = arith.addi %v, %b_ : i32
      linalg.yield %sum : i32
    } -> tensor<16x64xi32>
    %eR = tensor.empty() : tensor<16x64xi32>
    %relu = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%bias : tensor<16x64xi32>) outs(%eR : tensor<16x64xi32>) {
    ^bb0(%v: i32, %o: i32):
      %r = arith.maxsi %v, %z : i32
      linalg.yield %r : i32
    } -> tensor<16x64xi32>
    func.return %relu : tensor<16x64xi32>
  }
}
