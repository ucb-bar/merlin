builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf32>, %1: tensor<16xf32>, %2: tensor<16xf32>) -> tensor<16x16xf32> {
    %3 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %4 = tensor.splat %3 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %5 = linalg.reduce ins(%0:tensor<16x16xf32>) outs(%4:tensor<16xf32>) dimensions = [1]
    (%6: f32, %7: f32) {
      %8 = arith.addf %6, %7 : f32
      linalg.yield %8 : f32
    }
    %9 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} 1.600000e+01 : f32
    %10 = tensor.splat %9 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %11 = tensor.empty() : tensor<16xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %10 : tensor<16xf32>, tensor<16xf32>) outs(%11 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb0(%13: f32, %14: f32, %15: f32):
      %16 = arith.divf %13, %14 : f32
      linalg.yield %16 : f32
    } -> tensor<16xf32>
    %17 = tensor.expand_shape %12 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %18 = tensor.empty() : tensor<16x16xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %17 : tensor<16x16xf32>, tensor<16x1xf32>) outs(%18 : tensor<16x16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb1(%20: f32, %21: f32, %22: f32):
      %23 = arith.subf %20, %21 : f32
      linalg.yield %23 : f32
    } -> tensor<16x16xf32>
    %24 = tensor.empty() : tensor<16x16xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %19 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%24 : tensor<16x16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb2(%26: f32, %27: f32, %28: f32):
      %29 = arith.mulf %26, %27 : f32
      linalg.yield %29 : f32
    } -> tensor<16x16xf32>
    %30 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %31 = tensor.splat %30 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %32 = linalg.reduce ins(%25:tensor<16x16xf32>) outs(%31:tensor<16xf32>) dimensions = [1]
    (%33: f32, %34: f32) {
      %35 = arith.addf %33, %34 : f32
      linalg.yield %35 : f32
    }
    %36 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} 1.600000e+01 : f32
    %37 = tensor.splat %36 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %38 = tensor.empty() : tensor<16xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%32, %37 : tensor<16xf32>, tensor<16xf32>) outs(%38 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb3(%40: f32, %41: f32, %42: f32):
      %43 = arith.divf %40, %41 : f32
      linalg.yield %43 : f32
    } -> tensor<16xf32>
    %44 = tensor.expand_shape %39 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %45 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %46 = tensor.splat %45 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} : tensor<16x1xf32>
    %47 = tensor.empty() : tensor<16x1xf32>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %46 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%47 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb4(%49: f32, %50: f32, %51: f32):
      %52 = arith.addf %49, %50 : f32
      linalg.yield %52 : f32
    } -> tensor<16x1xf32>
    %53 = tensor.empty() : tensor<16x1xf32>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<16x1xf32>) outs(%53 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb5(%55: f32, %56: f32):
      %57 = math.rsqrt %55 : f32
      linalg.yield %57 : f32
    } -> tensor<16x1xf32>
    %58 = tensor.empty() : tensor<16x16xf32>
    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %54 : tensor<16x16xf32>, tensor<16x1xf32>) outs(%58 : tensor<16x16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb6(%60: f32, %61: f32, %62: f32):
      %63 = arith.mulf %60, %61 : f32
      linalg.yield %63 : f32
    } -> tensor<16x16xf32>
    %64 = tensor.empty() : tensor<16x16xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59, %1 : tensor<16x16xf32>, tensor<16xf32>) outs(%64 : tensor<16x16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb7(%66: f32, %67: f32, %68: f32):
      %69 = arith.mulf %66, %67 : f32
      linalg.yield %69 : f32
    } -> tensor<16x16xf32>
    %70 = tensor.empty() : tensor<16x16xf32>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %2 : tensor<16x16xf32>, tensor<16xf32>) outs(%70 : tensor<16x16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32"} {
    ^bb8(%72: f32, %73: f32, %74: f32):
      %75 = arith.addf %72, %73 : f32
      linalg.yield %75 : f32
    } -> tensor<16x16xf32>
    func.return %71 : tensor<16x16xf32>
  }
}
