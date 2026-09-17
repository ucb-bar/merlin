builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x31xf32>) -> tensor<16x31xf32> {
    %1 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %2 = tensor.splat %1 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %3 = linalg.reduce ins(%0:tensor<16x31xf32>) outs(%2:tensor<16xf32>) dimensions = [1]
    (%4: f32, %5: f32) {
      %6 = arith.maximumf %4, %5 : f32
      linalg.yield %6 : f32
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %8 = tensor.empty() : tensor<16x31xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %7 : tensor<16x31xf32>, tensor<16x1xf32>) outs(%8 : tensor<16x31xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb0(%10: f32, %11: f32, %12: f32):
      %13 = arith.subf %10, %11 : f32
      linalg.yield %13 : f32
    } -> tensor<16x31xf32>
    %14 = tensor.empty() : tensor<16x31xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<16x31xf32>) outs(%14 : tensor<16x31xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb1(%16: f32, %17: f32):
      %18 = math.exp %16 : f32
      linalg.yield %18 : f32
    } -> tensor<16x31xf32>
    %19 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %20 = tensor.splat %19 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %21 = linalg.reduce ins(%15:tensor<16x31xf32>) outs(%20:tensor<16xf32>) dimensions = [1]
    (%22: f32, %23: f32) {
      %24 = arith.addf %22, %23 : f32
      linalg.yield %24 : f32
    }
    %25 = tensor.expand_shape %21 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %26 = tensor.empty() : tensor<16x31xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %25 : tensor<16x31xf32>, tensor<16x1xf32>) outs(%26 : tensor<16x31xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%28: f32, %29: f32, %30: f32):
      %31 = arith.divf %28, %29 : f32
      linalg.yield %31 : f32
    } -> tensor<16x31xf32>
    func.return %27 : tensor<16x31xf32>
  }
}
