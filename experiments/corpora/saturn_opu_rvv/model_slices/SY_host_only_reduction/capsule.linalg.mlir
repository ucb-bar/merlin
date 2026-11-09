builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x64xf32>) -> tensor<32x1xf32> {
    %1 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %2 = tensor.splat %1 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %3 = linalg.reduce ins(%0:tensor<32x64xf32>) outs(%2:tensor<32xf32>) dimensions = [1]
    (%4: f32, %5: f32) {
      %6 = arith.addf %4, %5 : f32
      linalg.yield %6 : f32
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    func.return %7 : tensor<32x1xf32>
  }
}
