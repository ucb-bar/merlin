builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>) -> tensor<16x1xbf16> {
    %1 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %2 = tensor.splat %1 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %3 = linalg.reduce ins(%0:tensor<16x16xbf16>) outs(%2:tensor<16xbf16>) dimensions = [1]
    (%4: bf16, %5: bf16) {
      %6 = arith.addf %4, %5 : bf16
      linalg.yield %6 : bf16
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    func.return %7 : tensor<16x1xbf16>
  }
}
