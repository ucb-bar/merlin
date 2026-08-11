builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf16>, %1: tensor<1x16xf16>) -> tensor<16x16xf16> {
    %2 = tensor.empty() : tensor<16x16xf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xf16>) outs(%2 : tensor<16x16xf16>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float16"} {
    ^bb0(%4: f16, %5: f16):
      %6 = arith.constant 2.000000e+00 : f16
      %7 = math.powf %4, %6 : f16
      linalg.yield %7 : f16
    } -> tensor<16x16xf16>
    %8 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} 0.000000e+00 : f16
    %9 = tensor.splat %8 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} : tensor<16xf16>
    %10 = linalg.reduce ins(%3:tensor<16x16xf16>) outs(%9:tensor<16xf16>) dimensions = [1]
    (%11: f16, %12: f16) {
      %13 = arith.addf %11, %12 : f16
      linalg.yield %13 : f16
    }
    %14 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} 1.600000e+01 : f16
    %15 = tensor.splat %14 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} : tensor<16xf16>
    %16 = tensor.empty() : tensor<16xf16>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %15 : tensor<16xf16>, tensor<16xf16>) outs(%16 : tensor<16xf16>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} {
    ^bb1(%18: f16, %19: f16, %20: f16):
      %21 = arith.divf %18, %19 : f16
      linalg.yield %21 : f16
    } -> tensor<16xf16>
    %22 = tensor.expand_shape %17 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float16"} : tensor<16xf16> into tensor<16x1xf16>
    %23 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float16"} 1.001360e-05 : f16
    %24 = tensor.splat %23 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float16"} : tensor<16x1xf16>
    %25 = tensor.empty() : tensor<16x1xf16>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %24 : tensor<16x1xf16>, tensor<16x1xf16>) outs(%25 : tensor<16x1xf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float16"} {
    ^bb2(%27: f16, %28: f16, %29: f16):
      %30 = arith.addf %27, %28 : f16
      linalg.yield %30 : f16
    } -> tensor<16x1xf16>
    %31 = tensor.empty() : tensor<16x1xf16>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<16x1xf16>) outs(%31 : tensor<16x1xf16>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float16"} {
    ^bb3(%33: f16, %34: f16):
      %35 = math.rsqrt %33 : f16
      linalg.yield %35 : f16
    } -> tensor<16x1xf16>
    %36 = tensor.empty() : tensor<16x16xf16>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %32 : tensor<16x16xf16>, tensor<16x1xf16>) outs(%36 : tensor<16x16xf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float16"} {
    ^bb4(%38: f16, %39: f16, %40: f16):
      %41 = arith.mulf %38, %39 : f16
      linalg.yield %41 : f16
    } -> tensor<16x16xf16>
    %42 = tensor.empty() : tensor<16x16xf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %1 : tensor<16x16xf16>, tensor<1x16xf16>) outs(%42 : tensor<16x16xf16>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float16"} {
    ^bb5(%44: f16, %45: f16, %46: f16):
      %47 = arith.mulf %44, %45 : f16
      linalg.yield %47 : f16
    } -> tensor<16x16xf16>
    func.return %43 : tensor<16x16xf16>
  }
}
