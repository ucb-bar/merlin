// L3_conv2d_im2col_i8: conv2d NHWC/HWCF, i8 -> i32, unit stride, no padding
// (8x8x4 * 3x3 -> 6x6x8).
// Written as a linalg.generic with the conv access maps rather than a named op, which is what a
// QUANTISED torch export produces (the extsi widening is explicit). The im2col rewrite TASK_full
// mentions is one legal way to lower this; nothing here mandates it.
#cI = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh + kh, ow + kw, ic)>
#cW = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (kh, kw, ic, oc)>
#cO = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh, ow, oc)>
module attributes {merlin.capsule = "L3_conv2d_im2col_i8"} {
  func.func @forward(%IFM: tensor<1x8x8x4xi8> {merlin.role = "input"},
                     %W: tensor<3x3x4x8xi8> {merlin.role = "weight"}) -> tensor<1x6x6x8xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<1x6x6x8xi32>
    %init = linalg.fill ins(%z : i32) outs(%e : tensor<1x6x6x8xi32>) -> tensor<1x6x6x8xi32>
    %0 = linalg.generic {indexing_maps = [#cI, #cW, #cO],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel",
                                           "reduction", "reduction", "reduction"]}
         ins(%IFM, %W : tensor<1x8x8x4xi8>, tensor<3x3x4x8xi8>) outs(%init : tensor<1x6x6x8xi32>) {
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ea = arith.extsi %a : i8 to i32
      %eb = arith.extsi %b : i8 to i32
      %p = arith.muli %ea, %eb : i32
      %s = arith.addi %acc, %p : i32
      linalg.yield %s : i32
    } -> tensor<1x6x6x8xi32>
    func.return %0 : tensor<1x6x6x8xi32>
  }
}
