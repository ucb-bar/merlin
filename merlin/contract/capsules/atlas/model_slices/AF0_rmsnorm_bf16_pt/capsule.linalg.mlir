builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>, %1: tensor<1x16xbf16>) -> tensor<16x16xbf16> {
    %2 = tensor.empty() : tensor<16x16xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xbf16>) outs(%2 : tensor<16x16xbf16>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16):
      %6 = arith.constant 2.000000e+00 : bf16
      %7 = math.powf %4, %6 : bf16
      linalg.yield %7 : bf16
    } -> tensor<16x16xbf16>
    %8 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %9 = tensor.splat %8 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %10 = linalg.reduce ins(%3:tensor<16x16xbf16>) outs(%9:tensor<16xbf16>) dimensions = [1]
    (%11: bf16, %12: bf16) {
      %13 = arith.addf %11, %12 : bf16
      linalg.yield %13 : bf16
    }
    %14 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 1.600000e+01 : bf16
    %15 = tensor.splat %14 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %16 = tensor.empty() : tensor<16xbf16>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %15 : tensor<16xbf16>, tensor<16xbf16>) outs(%16 : tensor<16xbf16>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} {
    ^bb1(%18: bf16, %19: bf16, %20: bf16):
      %21 = arith.divf %18, %19 : bf16
      linalg.yield %21 : bf16
    } -> tensor<16xbf16>
    %22 = tensor.expand_shape %17 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %23 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 1.001360e-05 : bf16
    %24 = tensor.splat %23 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<16x1xbf16>
    %25 = tensor.empty() : tensor<16x1xbf16>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %24 : tensor<16x1xbf16>, tensor<16x1xbf16>) outs(%25 : tensor<16x1xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb2(%27: bf16, %28: bf16, %29: bf16):
      %30 = arith.addf %27, %28 : bf16
      linalg.yield %30 : bf16
    } -> tensor<16x1xbf16>
    %31 = tensor.empty() : tensor<16x1xbf16>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<16x1xbf16>) outs(%31 : tensor<16x1xbf16>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%33: bf16, %34: bf16):
      %35 = math.rsqrt %33 : bf16
      linalg.yield %35 : bf16
    } -> tensor<16x1xbf16>
    %36 = tensor.empty() : tensor<16x16xbf16>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %32 : tensor<16x16xbf16>, tensor<16x1xbf16>) outs(%36 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb4(%38: bf16, %39: bf16, %40: bf16):
      %41 = arith.mulf %38, %39 : bf16
      linalg.yield %41 : bf16
    } -> tensor<16x16xbf16>
    %42 = tensor.empty() : tensor<16x16xbf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %1 : tensor<16x16xbf16>, tensor<1x16xbf16>) outs(%42 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb5(%44: bf16, %45: bf16, %46: bf16):
      %47 = arith.mulf %44, %45 : bf16
      linalg.yield %47 : bf16
    } -> tensor<16x16xbf16>
    func.return %43 : tensor<16x16xbf16>
  }
}
