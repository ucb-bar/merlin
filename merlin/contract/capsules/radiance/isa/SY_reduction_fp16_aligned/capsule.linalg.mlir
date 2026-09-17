builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xf16>) -> tensor<16x1xf16> {
    %1 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float16"} 0.000000e+00 : f16
    %2 = tensor.splat %1 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float16"} : tensor<16xf16>
    %3 = linalg.reduce ins(%0:tensor<16x32xf16>) outs(%2:tensor<16xf16>) dimensions = [1]
    (%4: f16, %5: f16) {
      %6 = arith.addf %4, %5 : f16
      linalg.yield %6 : f16
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float16"} : tensor<16xf16> into tensor<16x1xf16>
    func.return %7 : tensor<16x1xf16>
  }
}
