builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x8x8x8xbf16>) -> tensor<1x8xbf16> {
    %1 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %2 = tensor.splat %1 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<1x8xbf16>
    %3 = linalg.reduce ins(%0:tensor<1x8x8x8xbf16>) outs(%2:tensor<1x8xbf16>) dimensions = [2, 3]
    (%4: bf16, %5: bf16) {
      %6 = arith.addf %4, %5 : bf16
      linalg.yield %6 : bf16
    }
    %7 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 6.400000e+01 : bf16
    %8 = tensor.splat %7 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<1x8xbf16>
    %9 = tensor.empty() : tensor<1x8xbf16>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %8 : tensor<1x8xbf16>, tensor<1x8xbf16>) outs(%9 : tensor<1x8xbf16>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} {
    ^bb0(%11: bf16, %12: bf16, %13: bf16):
      %14 = arith.divf %11, %12 : bf16
      linalg.yield %14 : bf16
    } -> tensor<1x8xbf16>
    func.return %10 : tensor<1x8xbf16>
  }
}
