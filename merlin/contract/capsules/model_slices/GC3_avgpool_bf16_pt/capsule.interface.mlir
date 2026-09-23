builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x8x8x8xbf16>) -> tensor<1x8x4x4xbf16> {
    %1 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1x8x8x8xbf16> into tensor<512xbf16>
    %2 = tensor.expand_shape %1 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 8, 4, 2, 4, 2] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<512xbf16> into tensor<1x8x4x2x4x2xbf16>
    %3 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %4 = tensor.splat %3 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<1x8x4x4xbf16>
    %5 = linalg.reduce ins(%2:tensor<1x8x4x2x4x2xbf16>) outs(%4:tensor<1x8x4x4xbf16>) dimensions = [3, 5]
    (%6: bf16, %7: bf16) {
      %8 = arith.addf %6, %7 : bf16
      linalg.yield %8 : bf16
    }
    %9 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 4.000000e+00 : bf16
    %10 = tensor.splat %9 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<1x8x4x4xbf16>
    %11 = tensor.empty() : tensor<1x8x4x4xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %10 : tensor<1x8x4x4xbf16>, tensor<1x8x4x4xbf16>) outs(%11 : tensor<1x8x4x4xbf16>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} {
    ^bb0(%13: bf16, %14: bf16, %15: bf16):
      %16 = arith.divf %13, %14 : bf16
      linalg.yield %16 : bf16
    } -> tensor<1x8x4x4xbf16>
    func.return %12 : tensor<1x8x4x4xbf16>
  }
}
