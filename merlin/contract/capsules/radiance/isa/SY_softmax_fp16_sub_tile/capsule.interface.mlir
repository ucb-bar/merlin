builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %1 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} 0xfc00 : f16
    %2 = tensor.splat %1 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<2xf16>
    %3 = linalg.reduce ins(%0:tensor<2x4xf16>) outs(%2:tensor<2xf16>) dimensions = [1]
    (%4: f16, %5: f16) {
      %6 = arith.maximumf %4, %5 : f16
      linalg.yield %6 : f16
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<2xf16> into tensor<2x1xf16>
    %8 = tensor.empty() : tensor<2x4xf16>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %7 : tensor<2x4xf16>, tensor<2x1xf16>) outs(%8 : tensor<2x4xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb0(%10: f16, %11: f16, %12: f16):
      %13 = arith.subf %10, %11 : f16
      linalg.yield %13 : f16
    } -> tensor<2x4xf16>
    %14 = tensor.empty() : tensor<2x4xf16>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2x4xf16>) outs(%14 : tensor<2x4xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb1(%16: f16, %17: f16):
      %18 = math.exp %16 : f16
      linalg.yield %18 : f16
    } -> tensor<2x4xf16>
    %19 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} 0.000000e+00 : f16
    %20 = tensor.splat %19 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<2xf16>
    %21 = linalg.reduce ins(%15:tensor<2x4xf16>) outs(%20:tensor<2xf16>) dimensions = [1]
    (%22: f16, %23: f16) {
      %24 = arith.addf %22, %23 : f16
      linalg.yield %24 : f16
    }
    %25 = tensor.expand_shape %21 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<2xf16> into tensor<2x1xf16>
    %26 = tensor.empty() : tensor<2x4xf16>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %25 : tensor<2x4xf16>, tensor<2x1xf16>) outs(%26 : tensor<2x4xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb2(%28: f16, %29: f16, %30: f16):
      %31 = arith.divf %28, %29 : f16
      linalg.yield %31 : f16
    } -> tensor<2x4xf16>
    func.return %27 : tensor<2x4xf16>
  }
}
