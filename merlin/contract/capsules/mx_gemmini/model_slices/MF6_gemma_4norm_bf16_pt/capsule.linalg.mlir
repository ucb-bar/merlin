builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>, %1: tensor<1x16xbf16>, %2: tensor<1x16xbf16>) -> tensor<16x16xbf16> {
    %3 = tensor.empty() : tensor<16x16xbf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xbf16>) outs(%3 : tensor<16x16xbf16>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "bfloat16"} {
    ^bb0(%5: bf16, %6: bf16):
      %7 = arith.constant 2.000000e+00 : bf16
      %8 = math.powf %5, %7 : bf16
      linalg.yield %8 : bf16
    } -> tensor<16x16xbf16>
    %9 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %10 = tensor.splat %9 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %11 = linalg.reduce ins(%4:tensor<16x16xbf16>) outs(%10:tensor<16xbf16>) dimensions = [1]
    (%12: bf16, %13: bf16) {
      %14 = arith.addf %12, %13 : bf16
      linalg.yield %14 : bf16
    }
    %15 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 1.600000e+01 : bf16
    %16 = tensor.splat %15 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %17 = tensor.empty() : tensor<16xbf16>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%11, %16 : tensor<16xbf16>, tensor<16xbf16>) outs(%17 : tensor<16xbf16>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} {
    ^bb1(%19: bf16, %20: bf16, %21: bf16):
      %22 = arith.divf %19, %20 : bf16
      linalg.yield %22 : bf16
    } -> tensor<16xbf16>
    %23 = tensor.expand_shape %18 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %24 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 1.001360e-05 : bf16
    %25 = tensor.splat %24 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<16x1xbf16>
    %26 = tensor.empty() : tensor<16x1xbf16>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : tensor<16x1xbf16>, tensor<16x1xbf16>) outs(%26 : tensor<16x1xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb2(%28: bf16, %29: bf16, %30: bf16):
      %31 = arith.addf %28, %29 : bf16
      linalg.yield %31 : bf16
    } -> tensor<16x1xbf16>
    %32 = tensor.empty() : tensor<16x1xbf16>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<16x1xbf16>) outs(%32 : tensor<16x1xbf16>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%34: bf16, %35: bf16):
      %36 = math.rsqrt %34 : bf16
      linalg.yield %36 : bf16
    } -> tensor<16x1xbf16>
    %37 = tensor.empty() : tensor<16x16xbf16>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %33 : tensor<16x16xbf16>, tensor<16x1xbf16>) outs(%37 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb4(%39: bf16, %40: bf16, %41: bf16):
      %42 = arith.mulf %39, %40 : bf16
      linalg.yield %42 : bf16
    } -> tensor<16x16xbf16>
    %43 = tensor.empty() : tensor<16x16xbf16>
    %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%38, %1 : tensor<16x16xbf16>, tensor<1x16xbf16>) outs(%43 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb5(%45: bf16, %46: bf16, %47: bf16):
      %48 = arith.mulf %45, %46 : bf16
      linalg.yield %48 : bf16
    } -> tensor<16x16xbf16>
    %49 = tensor.empty() : tensor<16x16xbf16>
    %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44 : tensor<16x16xbf16>) outs(%49 : tensor<16x16xbf16>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "bfloat16"} {
    ^bb6(%51: bf16, %52: bf16):
      %53 = arith.constant 2.000000e+00 : bf16
      %54 = math.powf %51, %53 : bf16
      linalg.yield %54 : bf16
    } -> tensor<16x16xbf16>
    %55 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %56 = tensor.splat %55 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %57 = linalg.reduce ins(%50:tensor<16x16xbf16>) outs(%56:tensor<16xbf16>) dimensions = [1]
    (%58: bf16, %59: bf16) {
      %60 = arith.addf %58, %59 : bf16
      linalg.yield %60 : bf16
    }
    %61 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} 1.600000e+01 : bf16
    %62 = tensor.splat %61 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %63 = tensor.empty() : tensor<16xbf16>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%57, %62 : tensor<16xbf16>, tensor<16xbf16>) outs(%63 : tensor<16xbf16>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} {
    ^bb7(%65: bf16, %66: bf16, %67: bf16):
      %68 = arith.divf %65, %66 : bf16
      linalg.yield %68 : bf16
    } -> tensor<16xbf16>
    %69 = tensor.expand_shape %64 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %70 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} 1.001360e-05 : bf16
    %71 = tensor.splat %70 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} : tensor<16x1xbf16>
    %72 = tensor.empty() : tensor<16x1xbf16>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %71 : tensor<16x1xbf16>, tensor<16x1xbf16>) outs(%72 : tensor<16x1xbf16>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb8(%74: bf16, %75: bf16, %76: bf16):
      %77 = arith.addf %74, %75 : bf16
      linalg.yield %77 : bf16
    } -> tensor<16x1xbf16>
    %78 = tensor.empty() : tensor<16x1xbf16>
    %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73 : tensor<16x1xbf16>) outs(%78 : tensor<16x1xbf16>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "bfloat16"} {
    ^bb9(%80: bf16, %81: bf16):
      %82 = math.rsqrt %80 : bf16
      linalg.yield %82 : bf16
    } -> tensor<16x1xbf16>
    %83 = tensor.empty() : tensor<16x16xbf16>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %79 : tensor<16x16xbf16>, tensor<16x1xbf16>) outs(%83 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb10(%85: bf16, %86: bf16, %87: bf16):
      %88 = arith.mulf %85, %86 : bf16
      linalg.yield %88 : bf16
    } -> tensor<16x16xbf16>
    %89 = tensor.empty() : tensor<16x16xbf16>
    %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%84, %2 : tensor<16x16xbf16>, tensor<1x16xbf16>) outs(%89 : tensor<16x16xbf16>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb11(%91: bf16, %92: bf16, %93: bf16):
      %94 = arith.mulf %91, %92 : bf16
      linalg.yield %94 : bf16
    } -> tensor<16x16xbf16>
    func.return %90 : tensor<16x16xbf16>
  }
}
