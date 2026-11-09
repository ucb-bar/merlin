builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
    %1 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %2 = tensor.splat %1 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<2xbf16>
    %3 = linalg.reduce ins(%0:tensor<2x4xbf16>) outs(%2:tensor<2xbf16>) dimensions = [1]
    (%4: bf16, %5: bf16) {
      %6 = arith.maximumf %4, %5 : bf16
      linalg.yield %6 : bf16
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<2xbf16> into tensor<2x1xbf16>
    %8 = tensor.empty() : tensor<2x4xbf16>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %7 : tensor<2x4xbf16>, tensor<2x1xbf16>) outs(%8 : tensor<2x4xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%10: bf16, %11: bf16, %12: bf16):
      %13 = arith.subf %10, %11 : bf16
      linalg.yield %13 : bf16
    } -> tensor<2x4xbf16>
    %14 = tensor.empty() : tensor<2x4xbf16>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2x4xbf16>) outs(%14 : tensor<2x4xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%16: bf16, %17: bf16):
      %18 = math.exp %16 : bf16
      linalg.yield %18 : bf16
    } -> tensor<2x4xbf16>
    %19 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %20 = tensor.splat %19 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<2xbf16>
    %21 = linalg.reduce ins(%15:tensor<2x4xbf16>) outs(%20:tensor<2xbf16>) dimensions = [1]
    (%22: bf16, %23: bf16) {
      %24 = arith.addf %22, %23 : bf16
      linalg.yield %24 : bf16
    }
    %25 = tensor.expand_shape %21 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<2xbf16> into tensor<2x1xbf16>
    %26 = tensor.empty() : tensor<2x4xbf16>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %25 : tensor<2x4xbf16>, tensor<2x1xbf16>) outs(%26 : tensor<2x4xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%28: bf16, %29: bf16, %30: bf16):
      %31 = arith.divf %28, %29 : bf16
      linalg.yield %31 : bf16
    } -> tensor<2x4xbf16>
    func.return %27 : tensor<2x4xbf16>
  }
}
