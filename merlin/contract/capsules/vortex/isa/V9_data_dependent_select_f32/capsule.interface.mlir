// V9_data_dependent_select_f32: a DATA-DEPENDENT branch — y = a > 0 ? a * 2 : -a.
// The sign of each element decides its arithmetic, and neighbouring lanes of a warp disagree, so the
// two arms cannot both be taken with a full thread mask. This is the capsule that requires actual
// reconvergence (split/join, or predication): stock LLVM emits a bare branch and inserts neither.
#ew = affine_map<(d0, d1) -> (d0, d1)>
module attributes {merlin.capsule = "V9_data_dependent_select_f32"} {
  func.func @forward(%A: tensor<8x8xf32> {merlin.role = "input"}) -> tensor<8x8xf32> {
    %z = arith.constant 0.000000e+00 : f32
    %two = arith.constant 2.000000e+00 : f32
    %e = tensor.empty() : tensor<8x8xf32>
    %0 = linalg.generic {indexing_maps = [#ew, #ew],
                         iterator_types = ["parallel", "parallel"]}
         ins(%A : tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) {
    ^bb0(%a: f32, %o: f32):
      %c = arith.cmpf ogt, %a, %z : f32
      %d = arith.mulf %a, %two : f32
      %n = arith.negf %a : f32
      %r = arith.select %c, %d, %n : f32
      linalg.yield %r : f32
    } -> tensor<8x8xf32>
    func.return %0 : tensor<8x8xf32>
  }
}
