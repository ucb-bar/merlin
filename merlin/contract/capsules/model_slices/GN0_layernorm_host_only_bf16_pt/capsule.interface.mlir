builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xbf16>, %1: tensor<32xbf16>, %2: tensor<32xbf16>) -> tensor<16x32xbf16> {
    %3 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %4 = tensor.splat %3 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %5 = linalg.reduce ins(%0:tensor<16x32xbf16>) outs(%4:tensor<16xbf16>) dimensions = [1]
    (%6: bf16, %7: bf16) {
      %8 = arith.addf %6, %7 : bf16
      linalg.yield %8 : bf16
    }
    %9 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} 3.200000e+01 : bf16
    %10 = tensor.splat %9 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %11 = tensor.empty() : tensor<16xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %10 : tensor<16xbf16>, tensor<16xbf16>) outs(%11 : tensor<16xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%13: bf16, %14: bf16, %15: bf16):
      %16 = arith.divf %13, %14 : bf16
      linalg.yield %16 : bf16
    } -> tensor<16xbf16>
    %17 = tensor.expand_shape %12 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %18 = tensor.empty() : tensor<16x32xbf16>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %17 : tensor<16x32xbf16>, tensor<16x1xbf16>) outs(%18 : tensor<16x32xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%20: bf16, %21: bf16, %22: bf16):
      %23 = arith.subf %20, %21 : bf16
      linalg.yield %23 : bf16
    } -> tensor<16x32xbf16>
    %24 = tensor.empty() : tensor<16x32xbf16>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %19 : tensor<16x32xbf16>, tensor<16x32xbf16>) outs(%24 : tensor<16x32xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%26: bf16, %27: bf16, %28: bf16):
      %29 = arith.mulf %26, %27 : bf16
      linalg.yield %29 : bf16
    } -> tensor<16x32xbf16>
    %30 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %31 = tensor.splat %30 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %32 = linalg.reduce ins(%25:tensor<16x32xbf16>) outs(%31:tensor<16xbf16>) dimensions = [1]
    (%33: bf16, %34: bf16) {
      %35 = arith.addf %33, %34 : bf16
      linalg.yield %35 : bf16
    }
    %36 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} 3.200000e+01 : bf16
    %37 = tensor.splat %36 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %38 = tensor.empty() : tensor<16xbf16>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%32, %37 : tensor<16xbf16>, tensor<16xbf16>) outs(%38 : tensor<16xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%40: bf16, %41: bf16, %42: bf16):
      %43 = arith.divf %40, %41 : bf16
      linalg.yield %43 : bf16
    } -> tensor<16xbf16>
    %44 = tensor.expand_shape %39 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %45 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} 1.525880e-05 : bf16
    %46 = tensor.splat %45 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} : tensor<16x1xbf16>
    %47 = tensor.empty() : tensor<16x1xbf16>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %46 : tensor<16x1xbf16>, tensor<16x1xbf16>) outs(%47 : tensor<16x1xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb4(%49: bf16, %50: bf16, %51: bf16):
      %52 = arith.addf %49, %50 : bf16
      linalg.yield %52 : bf16
    } -> tensor<16x1xbf16>
    %53 = tensor.empty() : tensor<16x1xbf16>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<16x1xbf16>) outs(%53 : tensor<16x1xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb5(%55: bf16, %56: bf16):
      %57 = math.rsqrt %55 : bf16
      linalg.yield %57 : bf16
    } -> tensor<16x1xbf16>
    %58 = tensor.empty() : tensor<16x32xbf16>
    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %54 : tensor<16x32xbf16>, tensor<16x1xbf16>) outs(%58 : tensor<16x32xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb6(%60: bf16, %61: bf16, %62: bf16):
      %63 = arith.mulf %60, %61 : bf16
      linalg.yield %63 : bf16
    } -> tensor<16x32xbf16>
    %64 = tensor.empty() : tensor<16x32xbf16>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59, %1 : tensor<16x32xbf16>, tensor<32xbf16>) outs(%64 : tensor<16x32xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb7(%66: bf16, %67: bf16, %68: bf16):
      %69 = arith.mulf %66, %67 : bf16
      linalg.yield %69 : bf16
    } -> tensor<16x32xbf16>
    %70 = tensor.empty() : tensor<16x32xbf16>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %2 : tensor<16x32xbf16>, tensor<32xbf16>) outs(%70 : tensor<16x32xbf16>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "bfloat16"} {
    ^bb8(%72: bf16, %73: bf16, %74: bf16):
      %75 = arith.addf %72, %73 : bf16
      linalg.yield %75 : bf16
    } -> tensor<16x32xbf16>
    func.return %71 : tensor<16x32xbf16>
  }
}
