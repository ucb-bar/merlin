builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x1x6x32xf32>, %1: tensor<1x1x1024x32xf32>, %2: tensor<1x1x1024x32xf32>) -> tensor<1x1x6x32xf32> {
    %3 = tensor.empty() : tensor<1x1x32x1024xf32>
    %4 = linalg.transpose ins(%1:tensor<1x1x1024x32xf32>) outs(%3:tensor<1x1x32x1024xf32>) permutation = [0, 1, 3, 2]
    %5 = tensor.empty() : tensor<1x1x6x32xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x6x32xf32>) outs(%5 : tensor<1x1x6x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb0(%7: f32, %8: f32):
      linalg.yield %7 : f32
    } -> tensor<1x1x6x32xf32>
    %9 = tensor.collapse_shape %6 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6x32xf32> into tensor<192xf32>
    %10 = tensor.expand_shape %9 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 32] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<192xf32> into tensor<1x6x32xf32>
    %11 = tensor.empty() : tensor<1x1x32x1024xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<1x1x32x1024xf32>) outs(%11 : tensor<1x1x32x1024xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb1(%13: f32, %14: f32):
      linalg.yield %13 : f32
    } -> tensor<1x1x32x1024xf32>
    %15 = tensor.empty() : tensor<1x1x1024x32xf32>
    %16 = linalg.transpose ins(%12:tensor<1x1x32x1024xf32>) outs(%15:tensor<1x1x1024x32xf32>) permutation = [0, 1, 3, 2]
    %17 = tensor.collapse_shape %16 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x1024x32xf32> into tensor<32768xf32>
    %18 = tensor.expand_shape %17 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1024, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x1024x32xf32>
    %19 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} 0x7f800000 : f32
    %20 = tensor.splat %19 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %21 = linalg.reduce ins(%10:tensor<1x6x32xf32>) outs(%20:tensor<1x6xf32>) dimensions = [2]
    (%22: f32, %23: f32) {
      %24 = arith.minimumf %22, %23 : f32
      linalg.yield %24 : f32
    }
    %25 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %26 = tensor.splat %25 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %27 = linalg.reduce ins(%10:tensor<1x6x32xf32>) outs(%26:tensor<1x6xf32>) dimensions = [2]
    (%28: f32, %29: f32) {
      %30 = arith.maximumf %28, %29 : f32
      linalg.yield %30 : f32
    }
    %31 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %32 = tensor.splat %31 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %33 = tensor.empty() : tensor<1x6xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %32 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%33 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "float32"} {
    ^bb2(%35: f32, %36: f32, %37: f32):
      %38 = arith.minimumf %35, %36 : f32
      linalg.yield %38 : f32
    } -> tensor<1x6xf32>
    %39 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %40 = tensor.splat %39 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %41 = tensor.empty() : tensor<1x6xf32>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%27, %40 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%41 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb3(%43: f32, %44: f32, %45: f32):
      %46 = arith.maximumf %43, %44 : f32
      linalg.yield %46 : f32
    } -> tensor<1x6xf32>
    %47 = tensor.empty() : tensor<1x6xf32>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<1x6xf32>) outs(%47 : tensor<1x6xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb4(%49: f32, %50: f32):
      %51 = arith.negf %49 : f32
      linalg.yield %51 : f32
    } -> tensor<1x6xf32>
    %52 = tensor.empty() : tensor<1x6xf32>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48, %42 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%52 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb5(%54: f32, %55: f32, %56: f32):
      %57 = arith.maximumf %54, %55 : f32
      linalg.yield %57 : f32
    } -> tensor<1x6xf32>
    %58 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.270000e+02 : f32
    %59 = tensor.splat %58 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %60 = tensor.empty() : tensor<1x6xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%53, %59 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%60 : tensor<1x6xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%62: f32, %63: f32, %64: f32):
      %65 = arith.divf %62, %63 : f32
      linalg.yield %65 : f32
    } -> tensor<1x6xf32>
    %66 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %67 = tensor.splat %66 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %68 = tensor.empty() : tensor<1x6xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61 : tensor<1x6xf32>) outs(%68 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb7(%70: f32, %71: f32):
      %72 = arith.constant 1.000000e-05 : f32
      %73 = arith.maximumf %70, %72 : f32
      linalg.yield %73 : f32
    } -> tensor<1x6xf32>
    %74 = tensor.collapse_shape %69 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %75 = tensor.expand_shape %74 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %76 = tensor.collapse_shape %67 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %77 = tensor.expand_shape %76 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %78 = tensor.empty() : tensor<1x6x1xf32>
    %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75 : tensor<1x6x1xf32>) outs(%78 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb8(%80: f32, %81: f32):
      %82 = arith.constant 1.000000e+00 : f32
      %83 = arith.divf %82, %80 : f32
      linalg.yield %83 : f32
    } -> tensor<1x6x1xf32>
    %84 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %85 = tensor.splat %84 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x6x1xf32>
    %86 = tensor.empty() : tensor<1x6x1xf32>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %85 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%86 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb9(%88: f32, %89: f32, %90: f32):
      %91 = arith.mulf %88, %89 : f32
      linalg.yield %91 : f32
    } -> tensor<1x6x1xf32>
    %92 = tensor.empty() : tensor<1x6x32xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %87 : tensor<1x6x32xf32>, tensor<1x6x1xf32>) outs(%92 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%94: f32, %95: f32, %96: f32):
      %97 = arith.mulf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<1x6x32xf32>
    %98 = tensor.empty() : tensor<1x6x32xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93 : tensor<1x6x32xf32>) outs(%98 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb11(%100: f32, %101: f32):
      %102 = math.roundeven %100 : f32
      linalg.yield %102 : f32
    } -> tensor<1x6x32xf32>
    %103 = tensor.empty() : tensor<1x6x32xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99, %77 : tensor<1x6x32xf32>, tensor<1x6x1xf32>) outs(%103 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb12(%105: f32, %106: f32, %107: f32):
      %108 = arith.addf %105, %106 : f32
      linalg.yield %108 : f32
    } -> tensor<1x6x32xf32>
    %109 = tensor.empty() : tensor<1x6x32xf32>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104 : tensor<1x6x32xf32>) outs(%109 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb13(%111: f32, %112: f32):
      %113 = arith.constant -1.270000e+02 : f32
      %114 = arith.maximumf %111, %113 : f32
      %115 = arith.constant 1.270000e+02 : f32
      %116 = arith.minimumf %114, %115 : f32
      linalg.yield %116 : f32
    } -> tensor<1x6x32xf32>
    %117 = tensor.empty() : tensor<1x6x32xi8>
    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%110 : tensor<1x6x32xf32>) outs(%117 : tensor<1x6x32xi8>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb14(%119: f32, %120: i8):
      %121 = arith.fptosi %119 : f32 to i8
      linalg.yield %121 : i8
    } -> tensor<1x6x32xi8>
    %122 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} 0x7f800000 : f32
    %123 = tensor.splat %122 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32>
    %124 = linalg.reduce ins(%18:tensor<1x1024x32xf32>) outs(%123:tensor<1x1024xf32>) dimensions = [2]
    (%125: f32, %126: f32) {
      %127 = arith.minimumf %125, %126 : f32
      linalg.yield %127 : f32
    }
    %128 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %129 = tensor.splat %128 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32>
    %130 = linalg.reduce ins(%18:tensor<1x1024x32xf32>) outs(%129:tensor<1x1024xf32>) dimensions = [2]
    (%131: f32, %132: f32) {
      %133 = arith.maximumf %131, %132 : f32
      linalg.yield %133 : f32
    }
    %134 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %135 = tensor.splat %134 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32>
    %136 = tensor.empty() : tensor<1x1024xf32>
    %137 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %135 : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%136 : tensor<1x1024xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "float32"} {
    ^bb15(%138: f32, %139: f32, %140: f32):
      %141 = arith.minimumf %138, %139 : f32
      linalg.yield %141 : f32
    } -> tensor<1x1024xf32>
    %142 = arith.constant {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %143 = tensor.splat %142 {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32>
    %144 = tensor.empty() : tensor<1x1024xf32>
    %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%130, %143 : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%144 : tensor<1x1024xf32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb16(%146: f32, %147: f32, %148: f32):
      %149 = arith.maximumf %146, %147 : f32
      linalg.yield %149 : f32
    } -> tensor<1x1024xf32>
    %150 = tensor.empty() : tensor<1x1024xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%137 : tensor<1x1024xf32>) outs(%150 : tensor<1x1024xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb17(%152: f32, %153: f32):
      %154 = arith.negf %152 : f32
      linalg.yield %154 : f32
    } -> tensor<1x1024xf32>
    %155 = tensor.empty() : tensor<1x1024xf32>
    %156 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%151, %145 : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%155 : tensor<1x1024xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb18(%157: f32, %158: f32, %159: f32):
      %160 = arith.maximumf %157, %158 : f32
      linalg.yield %160 : f32
    } -> tensor<1x1024xf32>
    %161 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.275000e+02 : f32
    %162 = tensor.splat %161 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x1024xf32>
    %163 = tensor.empty() : tensor<1x1024xf32>
    %164 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%156, %162 : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%163 : tensor<1x1024xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb19(%165: f32, %166: f32, %167: f32):
      %168 = arith.divf %165, %166 : f32
      linalg.yield %168 : f32
    } -> tensor<1x1024xf32>
    %169 = tensor.empty() : tensor<1x1024xf32>
    %170 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%164 : tensor<1x1024xf32>) outs(%169 : tensor<1x1024xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb20(%171: f32, %172: f32):
      %173 = arith.constant 1.1920929e-07 : f32
      %174 = arith.maximumf %171, %173 : f32
      linalg.yield %174 : f32
    } -> tensor<1x1024xf32>
    %175 = tensor.collapse_shape %170 [[0 : i64, 1 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
    %176 = tensor.expand_shape %175 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1024, 1] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1024x1xf32>
    %177 = tensor.empty() : tensor<1x1024x1xf32>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176 : tensor<1x1024x1xf32>) outs(%177 : tensor<1x1024x1xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb21(%179: f32, %180: f32):
      %181 = arith.constant 1.000000e+00 : f32
      %182 = arith.divf %181, %179 : f32
      linalg.yield %182 : f32
    } -> tensor<1x1024x1xf32>
    %183 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %184 = tensor.splat %183 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1024x1xf32>
    %185 = tensor.empty() : tensor<1x1024x1xf32>
    %186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178, %184 : tensor<1x1024x1xf32>, tensor<1x1024x1xf32>) outs(%185 : tensor<1x1024x1xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb22(%187: f32, %188: f32, %189: f32):
      %190 = arith.mulf %187, %188 : f32
      linalg.yield %190 : f32
    } -> tensor<1x1024x1xf32>
    %191 = tensor.empty() : tensor<1x1024x32xf32>
    %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %186 : tensor<1x1024x32xf32>, tensor<1x1024x1xf32>) outs(%191 : tensor<1x1024x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb23(%193: f32, %194: f32, %195: f32):
      %196 = arith.mulf %193, %194 : f32
      linalg.yield %196 : f32
    } -> tensor<1x1024x32xf32>
    %197 = tensor.empty() : tensor<1x1024x32xf32>
    %198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192 : tensor<1x1024x32xf32>) outs(%197 : tensor<1x1024x32xf32>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb24(%199: f32, %200: f32):
      %201 = math.roundeven %199 : f32
      linalg.yield %201 : f32
    } -> tensor<1x1024x32xf32>
    %202 = tensor.empty() : tensor<1x1024x32xf32>
    %203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x1024x32xf32>) outs(%202 : tensor<1x1024x32xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb25(%204: f32, %205: f32):
      %206 = arith.constant -1.280000e+02 : f32
      %207 = arith.maximumf %204, %206 : f32
      %208 = arith.constant 1.270000e+02 : f32
      %209 = arith.minimumf %207, %208 : f32
      linalg.yield %209 : f32
    } -> tensor<1x1024x32xf32>
    %210 = tensor.empty() : tensor<1x1024x32xi8>
    %211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%203 : tensor<1x1024x32xf32>) outs(%210 : tensor<1x1024x32xi8>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb26(%212: f32, %213: i8):
      %214 = arith.fptosi %212 : f32 to i8
      linalg.yield %214 : i8
    } -> tensor<1x1024x32xi8>
    %215 = "tensor.extract_slice"(%118) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 6, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<1x6x32xi8>) -> tensor<1x6x32xi8>
    %216 = tensor.collapse_shape %215 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x6x32xi8> into tensor<192xi8>
    %217 = tensor.expand_shape %216 [[0 : i64, 1 : i64]] output_shape [6, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<192xi8> into tensor<6x32xi8>
    %218 = "tensor.extract_slice"(%211) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1024, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<1x1024x32xi8>) -> tensor<1x1024x32xi8>
    %219 = tensor.collapse_shape %218 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x1024x32xi8> into tensor<32768xi8>
    %220 = tensor.expand_shape %219 [[0 : i64, 1 : i64]] output_shape [1024, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<32768xi8> into tensor<1024x32xi8>
    %221 = tensor.empty() : tensor<32x1024xi8>
    %222 = linalg.transpose ins(%220:tensor<1024x32xi8>) outs(%221:tensor<32x1024xi8>) permutation = [1, 0]
    %223 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %224 = tensor.splat %223 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<6x1024xi32>
    %225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%217, %222 : tensor<6x32xi8>, tensor<32x1024xi8>) outs(%224 : tensor<6x1024xi32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb27(%226: i8, %227: i8, %228: i32):
      %229 = arith.extsi %226 : i8 to i32
      %230 = arith.extsi %227 : i8 to i32
      %231 = arith.muli %229, %230 : i32
      %232 = arith.addi %228, %231 : i32
      linalg.yield %232 : i32
    } -> tensor<6x1024xi32>
    %233 = tensor.concat dim(0) %225 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int32"} : (tensor<6x1024xi32>) -> tensor<6x1024xi32>
    %234 = tensor.collapse_shape %233 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<6x1024xi32> into tensor<6144xi32>
    %235 = tensor.expand_shape %234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<6144xi32> into tensor<1x6x1024xi32>
    %236 = tensor.empty() : tensor<1x6x1024xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%235 : tensor<1x6x1024xi32>) outs(%236 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb28(%238: i32, %239: f32):
      %240 = arith.sitofp %238 : i32 to f32
      linalg.yield %240 : f32
    } -> tensor<1x6x1024xf32>
    %241 = tensor.collapse_shape %69 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %242 = tensor.expand_shape %241 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %243 = tensor.empty() : tensor<1x6x1024xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%237, %242 : tensor<1x6x1024xf32>, tensor<1x6x1xf32>) outs(%243 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb29(%245: f32, %246: f32, %247: f32):
      %248 = arith.mulf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x6x1024xf32>
    %249 = tensor.collapse_shape %170 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
    %250 = tensor.expand_shape %249 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
    %251 = tensor.empty() : tensor<1x6x1024xf32>
    %252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%244, %250 : tensor<1x6x1024xf32>, tensor<1x1x1024xf32>) outs(%251 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb30(%253: f32, %254: f32, %255: f32):
      %256 = arith.mulf %253, %254 : f32
      linalg.yield %256 : f32
    } -> tensor<1x6x1024xf32>
    %257 = tensor.collapse_shape %252 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6x1024xf32> into tensor<6144xf32>
    %258 = tensor.expand_shape %257 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 6, 1024] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x1x6x1024xf32>
    %259 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %260 = tensor.splat %259 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x6x1024xf32>
    %261 = tensor.empty() : tensor<1x1x6x1024xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%258, %260 : tensor<1x1x6x1024xf32>, tensor<1x1x6x1024xf32>) outs(%261 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb31(%263: f32, %264: f32, %265: f32):
      %266 = arith.mulf %263, %264 : f32
      linalg.yield %266 : f32
    } -> tensor<1x1x6x1024xf32>
    %267 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %268 = tensor.splat %267 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x1x6xf32>
    %269 = linalg.reduce ins(%262:tensor<1x1x6x1024xf32>) outs(%268:tensor<1x1x6xf32>) dimensions = [3]
    (%270: f32, %271: f32) {
      %272 = arith.maximumf %270, %271 : f32
      linalg.yield %272 : f32
    }
    %273 = tensor.collapse_shape %269 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x1x6xf32> into tensor<6xf32>
    %274 = tensor.expand_shape %273 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 6, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x1x6x1xf32>
    %275 = tensor.empty() : tensor<1x1x6x1024xf32>
    %276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%262, %274 : tensor<1x1x6x1024xf32>, tensor<1x1x6x1xf32>) outs(%275 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb32(%277: f32, %278: f32, %279: f32):
      %280 = arith.subf %277, %278 : f32
      linalg.yield %280 : f32
    } -> tensor<1x1x6x1024xf32>
    %281 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 0.00270760618 : f32
    %282 = tensor.splat %281 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x6x1024xf32>
    %283 = tensor.empty() : tensor<1x1x6x1024xf32>
    %284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %282 : tensor<1x1x6x1024xf32>, tensor<1x1x6x1024xf32>) outs(%283 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb33(%285: f32, %286: f32, %287: f32):
      %288 = arith.divf %285, %286 : f32
      linalg.yield %288 : f32
    } -> tensor<1x1x6x1024xf32>
    %289 = tensor.empty() : tensor<1x1x6x1024xf32>
    %290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284 : tensor<1x1x6x1024xf32>) outs(%289 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb34(%291: f32, %292: f32):
      %293 = math.roundeven %291 : f32
      linalg.yield %293 : f32
    } -> tensor<1x1x6x1024xf32>
    %294 = tensor.empty() : tensor<1x1x6x1024xf32>
    %295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290 : tensor<1x1x6x1024xf32>) outs(%294 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb35(%296: f32, %297: f32):
      %298 = arith.constant -1.108000e+04 : f32
      %299 = arith.maximumf %296, %298 : f32
      %300 = arith.constant 0.000000e+00 : f32
      %301 = arith.minimumf %299, %300 : f32
      linalg.yield %301 : f32
    } -> tensor<1x1x6x1024xf32>
    %302 = tensor.empty() : tensor<1x1x6x1024xi32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%295 : tensor<1x1x6x1024xf32>) outs(%302 : tensor<1x1x6x1024xi32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb36(%304: f32, %305: i32):
      %306 = arith.fptosi %304 : f32 to i32
      linalg.yield %306 : i32
    } -> tensor<1x1x6x1024xi32>
    %307 = tensor.empty() : tensor<1x1x6x1024xi64>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%303 : tensor<1x1x6x1024xi32>) outs(%307 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int64"} {
    ^bb37(%309: i32, %310: i64):
      %311 = arith.extsi %309 : i32 to i64
      linalg.yield %311 : i64
    } -> tensor<1x1x6x1024xi64>
    %312 = arith.constant {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} 0 : i64
    %313 = tensor.splat %312 {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %314 = tensor.empty() : tensor<1x1x6x1024xi64>
    %315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%313, %308 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%314 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} {
    ^bb38(%316: i64, %317: i64, %318: i64):
      %319 = arith.subi %316, %317 : i64
      linalg.yield %319 : i64
    } -> tensor<1x1x6x1024xi64>
    %320 = tensor.empty() : tensor<1x1x6x1024xi64>
    %321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%315 : tensor<1x1x6x1024xi64>) outs(%320 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor_Scalar", prov.orig_dtype = "int64"} {
    ^bb39(%322: i64, %323: i64):
      %324 = arith.constant 8 : i64
      %325 = arith.shrsi %322, %324 : i64
      linalg.yield %325 : i64
    } -> tensor<1x1x6x1024xi64>
    %326 = arith.constant {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 256 : i64
    %327 = tensor.splat %326 {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %328 = tensor.empty() : tensor<1x1x6x1024xi64>
    %329 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%321, %327 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%328 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb40(%330: i64, %331: i64, %332: i64):
      %333 = arith.muli %330, %331 : i64
      linalg.yield %333 : i64
    } -> tensor<1x1x6x1024xi64>
    %334 = tensor.empty() : tensor<1x1x6x1024xi64>
    %335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308, %329 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%334 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb41(%336: i64, %337: i64, %338: i64):
      %339 = arith.addi %336, %337 : i64
      linalg.yield %339 : i64
    } -> tensor<1x1x6x1024xi64>
    %340 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 500 : i64
    %341 = tensor.splat %340 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %342 = tensor.empty() : tensor<1x1x6x1024xi64>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%335, %341 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%342 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb42(%344: i64, %345: i64, %346: i64):
      %347 = arith.addi %344, %345 : i64
      linalg.yield %347 : i64
    } -> tensor<1x1x6x1024xi64>
    %348 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 2819 : i64
    %349 = tensor.splat %348 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %350 = tensor.empty() : tensor<1x1x6x1024xi64>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %349 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%350 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb43(%352: i64, %353: i64, %354: i64):
      %355 = arith.muli %352, %353 : i64
      linalg.yield %355 : i64
    } -> tensor<1x1x6x1024xi64>
    %356 = tensor.empty() : tensor<1x1x6x1024xi64>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351, %343 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%356 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb44(%358: i64, %359: i64, %360: i64):
      %361 = arith.muli %358, %359 : i64
      linalg.yield %361 : i64
    } -> tensor<1x1x6x1024xi64>
    %362 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 369388222 : i64
    %363 = tensor.splat %362 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %364 = tensor.empty() : tensor<1x1x6x1024xi64>
    %365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%357, %363 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%364 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb45(%366: i64, %367: i64, %368: i64):
      %369 = arith.addi %366, %367 : i64
      linalg.yield %369 : i64
    } -> tensor<1x1x6x1024xi64>
    %370 = tensor.empty() : tensor<1x1x6x1024xi64>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%365, %321 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%370 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor", prov.orig_dtype = "int64"} {
    ^bb46(%372: i64, %373: i64, %374: i64):
      %375 = arith.constant 63 : i64
      %376 = arith.minui %373, %375 : i64
      %377 = arith.shrsi %372, %376 : i64
      linalg.yield %377 : i64
    } -> tensor<1x1x6x1024xi64>
    %378 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 127 : i64
    %379 = tensor.splat %378 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %380 = tensor.empty() : tensor<1x1x6x1024xi64>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%371, %379 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%380 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb47(%382: i64, %383: i64, %384: i64):
      %385 = arith.muli %382, %383 : i64
      linalg.yield %385 : i64
    } -> tensor<1x1x6x1024xi64>
    %386 = arith.constant {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 537069111 : i64
    %387 = tensor.splat %386 {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x1x6x1024xi64>
    %388 = tensor.empty() : tensor<1x1x6x1024xi64>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %387 : tensor<1x1x6x1024xi64>, tensor<1x1x6x1024xi64>) outs(%388 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb48(%390: i64, %391: i64, %392: i64):
      %393 = arith.addi %390, %391 : i64
      linalg.yield %393 : i64
    } -> tensor<1x1x6x1024xi64>
    %394 = tensor.empty() : tensor<1x1x6x1024xi64>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389 : tensor<1x1x6x1024xi64>) outs(%394 : tensor<1x1x6x1024xi64>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "floor_divide", prov.op = "floor_divide", prov.aten = "aten.div.Tensor_mode", prov.orig_dtype = "int64"} {
    ^bb49(%396: i64, %397: i64):
      %398 = arith.constant 1074138222 : i64
      %399 = arith.floordivsi %396, %398 : i64
      linalg.yield %399 : i64
    } -> tensor<1x1x6x1024xi64>
    %400 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} 0 : i64
    %401 = tensor.splat %400 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<1x1x6xi64>
    %402 = linalg.reduce ins(%395:tensor<1x1x6x1024xi64>) outs(%401:tensor<1x1x6xi64>) dimensions = [3]
    (%403: i64, %404: i64) {
      %405 = arith.addi %403, %404 : i64
      linalg.yield %405 : i64
    }
    %406 = tensor.collapse_shape %402 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<1x1x6xi64> into tensor<6xi64>
    %407 = tensor.expand_shape %406 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 6, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<6xi64> into tensor<1x1x6x1xi64>
    %408 = tensor.empty() : tensor<1x1x6x1024xf32>
    %409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395 : tensor<1x1x6x1024xi64>) outs(%408 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb50(%410: i64, %411: f32):
      %412 = arith.sitofp %410 : i64 to f32
      linalg.yield %412 : f32
    } -> tensor<1x1x6x1024xf32>
    %413 = tensor.empty() : tensor<1x1x6x1xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407 : tensor<1x1x6x1xi64>) outs(%413 : tensor<1x1x6x1xf32>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb51(%415: i64, %416: f32):
      %417 = arith.sitofp %415 : i64 to f32
      linalg.yield %417 : f32
    } -> tensor<1x1x6x1xf32>
    %418 = tensor.empty() : tensor<1x1x6x1024xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %414 : tensor<1x1x6x1024xf32>, tensor<1x1x6x1xf32>) outs(%418 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb52(%420: f32, %421: f32, %422: f32):
      %423 = arith.divf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x1x6x1024xf32>
    %424 = tensor.empty() : tensor<1x1x6x1024xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419 : tensor<1x1x6x1024xf32>) outs(%424 : tensor<1x1x6x1024xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb53(%426: f32, %427: f32):
      linalg.yield %426 : f32
    } -> tensor<1x1x6x1024xf32>
    %428 = tensor.collapse_shape %425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6x1024xf32> into tensor<6144xf32>
    %429 = tensor.expand_shape %428 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x6x1024xf32>
    %430 = tensor.empty() : tensor<1x1x1024x32xf32>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1x1024x32xf32>) outs(%430 : tensor<1x1x1024x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb54(%432: f32, %433: f32):
      linalg.yield %432 : f32
    } -> tensor<1x1x1024x32xf32>
    %434 = tensor.empty() : tensor<1x1x32x1024xf32>
    %435 = linalg.transpose ins(%431:tensor<1x1x1024x32xf32>) outs(%434:tensor<1x1x32x1024xf32>) permutation = [0, 1, 3, 2]
    %436 = tensor.collapse_shape %435 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x32x1024xf32> into tensor<32768xf32>
    %437 = tensor.expand_shape %436 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x1024xf32>
    %438 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} 0x7f800000 : f32
    %439 = tensor.splat %438 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %440 = linalg.reduce ins(%429:tensor<1x6x1024xf32>) outs(%439:tensor<1x6xf32>) dimensions = [2]
    (%441: f32, %442: f32) {
      %443 = arith.minimumf %441, %442 : f32
      linalg.yield %443 : f32
    }
    %444 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %445 = tensor.splat %444 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %446 = linalg.reduce ins(%429:tensor<1x6x1024xf32>) outs(%445:tensor<1x6xf32>) dimensions = [2]
    (%447: f32, %448: f32) {
      %449 = arith.maximumf %447, %448 : f32
      linalg.yield %449 : f32
    }
    %450 = arith.constant {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %451 = tensor.splat %450 {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %452 = tensor.empty() : tensor<1x6xf32>
    %453 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%440, %451 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%452 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "float32"} {
    ^bb55(%454: f32, %455: f32, %456: f32):
      %457 = arith.minimumf %454, %455 : f32
      linalg.yield %457 : f32
    } -> tensor<1x6xf32>
    %458 = arith.constant {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %459 = tensor.splat %458 {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %460 = tensor.empty() : tensor<1x6xf32>
    %461 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%446, %459 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%460 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb56(%462: f32, %463: f32, %464: f32):
      %465 = arith.maximumf %462, %463 : f32
      linalg.yield %465 : f32
    } -> tensor<1x6xf32>
    %466 = tensor.empty() : tensor<1x6xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%453 : tensor<1x6xf32>) outs(%466 : tensor<1x6xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb57(%468: f32, %469: f32):
      %470 = arith.negf %468 : f32
      linalg.yield %470 : f32
    } -> tensor<1x6xf32>
    %471 = tensor.empty() : tensor<1x6xf32>
    %472 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%467, %461 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%471 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb58(%473: f32, %474: f32, %475: f32):
      %476 = arith.maximumf %473, %474 : f32
      linalg.yield %476 : f32
    } -> tensor<1x6xf32>
    %477 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.270000e+02 : f32
    %478 = tensor.splat %477 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %479 = tensor.empty() : tensor<1x6xf32>
    %480 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%472, %478 : tensor<1x6xf32>, tensor<1x6xf32>) outs(%479 : tensor<1x6xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb59(%481: f32, %482: f32, %483: f32):
      %484 = arith.divf %481, %482 : f32
      linalg.yield %484 : f32
    } -> tensor<1x6xf32>
    %485 = arith.constant {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %486 = tensor.splat %485 {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x6xf32>
    %487 = tensor.empty() : tensor<1x6xf32>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%480 : tensor<1x6xf32>) outs(%487 : tensor<1x6xf32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb60(%489: f32, %490: f32):
      %491 = arith.constant 1.000000e-05 : f32
      %492 = arith.maximumf %489, %491 : f32
      linalg.yield %492 : f32
    } -> tensor<1x6xf32>
    %493 = tensor.collapse_shape %488 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %494 = tensor.expand_shape %493 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %495 = tensor.collapse_shape %486 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %496 = tensor.expand_shape %495 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %497 = tensor.empty() : tensor<1x6x1xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%494 : tensor<1x6x1xf32>) outs(%497 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb61(%499: f32, %500: f32):
      %501 = arith.constant 1.000000e+00 : f32
      %502 = arith.divf %501, %499 : f32
      linalg.yield %502 : f32
    } -> tensor<1x6x1xf32>
    %503 = arith.constant {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %504 = tensor.splat %503 {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x6x1xf32>
    %505 = tensor.empty() : tensor<1x6x1xf32>
    %506 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%498, %504 : tensor<1x6x1xf32>, tensor<1x6x1xf32>) outs(%505 : tensor<1x6x1xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb62(%507: f32, %508: f32, %509: f32):
      %510 = arith.mulf %507, %508 : f32
      linalg.yield %510 : f32
    } -> tensor<1x6x1xf32>
    %511 = tensor.empty() : tensor<1x6x1024xf32>
    %512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%429, %506 : tensor<1x6x1024xf32>, tensor<1x6x1xf32>) outs(%511 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb63(%513: f32, %514: f32, %515: f32):
      %516 = arith.mulf %513, %514 : f32
      linalg.yield %516 : f32
    } -> tensor<1x6x1024xf32>
    %517 = tensor.empty() : tensor<1x6x1024xf32>
    %518 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%512 : tensor<1x6x1024xf32>) outs(%517 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb64(%519: f32, %520: f32):
      %521 = math.roundeven %519 : f32
      linalg.yield %521 : f32
    } -> tensor<1x6x1024xf32>
    %522 = tensor.empty() : tensor<1x6x1024xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%518, %496 : tensor<1x6x1024xf32>, tensor<1x6x1xf32>) outs(%522 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb65(%524: f32, %525: f32, %526: f32):
      %527 = arith.addf %524, %525 : f32
      linalg.yield %527 : f32
    } -> tensor<1x6x1024xf32>
    %528 = tensor.empty() : tensor<1x6x1024xf32>
    %529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%523 : tensor<1x6x1024xf32>) outs(%528 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb66(%530: f32, %531: f32):
      %532 = arith.constant -1.270000e+02 : f32
      %533 = arith.maximumf %530, %532 : f32
      %534 = arith.constant 1.270000e+02 : f32
      %535 = arith.minimumf %533, %534 : f32
      linalg.yield %535 : f32
    } -> tensor<1x6x1024xf32>
    %536 = tensor.empty() : tensor<1x6x1024xi8>
    %537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%529 : tensor<1x6x1024xf32>) outs(%536 : tensor<1x6x1024xi8>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb67(%538: f32, %539: i8):
      %540 = arith.fptosi %538 : f32 to i8
      linalg.yield %540 : i8
    } -> tensor<1x6x1024xi8>
    %541 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} 0x7f800000 : f32
    %542 = tensor.splat %541 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %543 = linalg.reduce ins(%437:tensor<1x32x1024xf32>) outs(%542:tensor<1x32xf32>) dimensions = [2]
    (%544: f32, %545: f32) {
      %546 = arith.minimumf %544, %545 : f32
      linalg.yield %546 : f32
    }
    %547 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %548 = tensor.splat %547 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %549 = linalg.reduce ins(%437:tensor<1x32x1024xf32>) outs(%548:tensor<1x32xf32>) dimensions = [2]
    (%550: f32, %551: f32) {
      %552 = arith.maximumf %550, %551 : f32
      linalg.yield %552 : f32
    }
    %553 = arith.constant {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %554 = tensor.splat %553 {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %555 = tensor.empty() : tensor<1x32xf32>
    %556 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%543, %554 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%555 : tensor<1x32xf32>) attrs =  {prov.region_id = "minmax_16", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "float32"} {
    ^bb68(%557: f32, %558: f32, %559: f32):
      %560 = arith.minimumf %557, %558 : f32
      linalg.yield %560 : f32
    } -> tensor<1x32xf32>
    %561 = arith.constant {prov.region_id = "fill_9", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %562 = tensor.splat %561 {prov.region_id = "fill_9", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %563 = tensor.empty() : tensor<1x32xf32>
    %564 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%549, %562 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%563 : tensor<1x32xf32>) attrs =  {prov.region_id = "minmax_17", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb69(%565: f32, %566: f32, %567: f32):
      %568 = arith.maximumf %565, %566 : f32
      linalg.yield %568 : f32
    } -> tensor<1x32xf32>
    %569 = tensor.empty() : tensor<1x32xf32>
    %570 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%556 : tensor<1x32xf32>) outs(%569 : tensor<1x32xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb70(%571: f32, %572: f32):
      %573 = arith.negf %571 : f32
      linalg.yield %573 : f32
    } -> tensor<1x32xf32>
    %574 = tensor.empty() : tensor<1x32xf32>
    %575 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%570, %564 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%574 : tensor<1x32xf32>) attrs =  {prov.region_id = "minmax_18", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "float32"} {
    ^bb71(%576: f32, %577: f32, %578: f32):
      %579 = arith.maximumf %576, %577 : f32
      linalg.yield %579 : f32
    } -> tensor<1x32xf32>
    %580 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.275000e+02 : f32
    %581 = tensor.splat %580 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %582 = tensor.empty() : tensor<1x32xf32>
    %583 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%575, %581 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%582 : tensor<1x32xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb72(%584: f32, %585: f32, %586: f32):
      %587 = arith.divf %584, %585 : f32
      linalg.yield %587 : f32
    } -> tensor<1x32xf32>
    %588 = tensor.empty() : tensor<1x32xf32>
    %589 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%583 : tensor<1x32xf32>) outs(%588 : tensor<1x32xf32>) attrs =  {prov.region_id = "minmax_19", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb73(%590: f32, %591: f32):
      %592 = arith.constant 1.1920929e-07 : f32
      %593 = arith.maximumf %590, %592 : f32
      linalg.yield %593 : f32
    } -> tensor<1x32xf32>
    %594 = tensor.collapse_shape %589 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %595 = tensor.expand_shape %594 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
    %596 = tensor.empty() : tensor<1x32x1xf32>
    %597 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%595 : tensor<1x32x1xf32>) outs(%596 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb74(%598: f32, %599: f32):
      %600 = arith.constant 1.000000e+00 : f32
      %601 = arith.divf %600, %598 : f32
      linalg.yield %601 : f32
    } -> tensor<1x32x1xf32>
    %602 = arith.constant {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %603 = tensor.splat %602 {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
    %604 = tensor.empty() : tensor<1x32x1xf32>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597, %603 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%604 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb75(%606: f32, %607: f32, %608: f32):
      %609 = arith.mulf %606, %607 : f32
      linalg.yield %609 : f32
    } -> tensor<1x32x1xf32>
    %610 = tensor.empty() : tensor<1x32x1024xf32>
    %611 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%437, %605 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%610 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb76(%612: f32, %613: f32, %614: f32):
      %615 = arith.mulf %612, %613 : f32
      linalg.yield %615 : f32
    } -> tensor<1x32x1024xf32>
    %616 = tensor.empty() : tensor<1x32x1024xf32>
    %617 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%611 : tensor<1x32x1024xf32>) outs(%616 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb77(%618: f32, %619: f32):
      %620 = math.roundeven %618 : f32
      linalg.yield %620 : f32
    } -> tensor<1x32x1024xf32>
    %621 = tensor.empty() : tensor<1x32x1024xf32>
    %622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%617 : tensor<1x32x1024xf32>) outs(%621 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "minmax_20", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb78(%623: f32, %624: f32):
      %625 = arith.constant -1.280000e+02 : f32
      %626 = arith.maximumf %623, %625 : f32
      %627 = arith.constant 1.270000e+02 : f32
      %628 = arith.minimumf %626, %627 : f32
      linalg.yield %628 : f32
    } -> tensor<1x32x1024xf32>
    %629 = tensor.empty() : tensor<1x32x1024xi8>
    %630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%622 : tensor<1x32x1024xf32>) outs(%629 : tensor<1x32x1024xi8>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb79(%631: f32, %632: i8):
      %633 = arith.fptosi %631 : f32 to i8
      linalg.yield %633 : i8
    } -> tensor<1x32x1024xi8>
    %634 = "tensor.extract_slice"(%537) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 6, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<1x6x1024xi8>) -> tensor<1x6x1024xi8>
    %635 = tensor.collapse_shape %634 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x6x1024xi8> into tensor<6144xi8>
    %636 = tensor.expand_shape %635 [[0 : i64, 1 : i64]] output_shape [6, 1024] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<6144xi8> into tensor<6x1024xi8>
    %637 = "tensor.extract_slice"(%630) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<1x32x1024xi8>) -> tensor<1x32x1024xi8>
    %638 = tensor.collapse_shape %637 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x32x1024xi8> into tensor<32768xi8>
    %639 = tensor.expand_shape %638 [[0 : i64, 1 : i64]] output_shape [32, 1024] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<32768xi8> into tensor<32x1024xi8>
    %640 = tensor.empty() : tensor<1024x32xi8>
    %641 = linalg.transpose ins(%639:tensor<32x1024xi8>) outs(%640:tensor<1024x32xi8>) permutation = [1, 0]
    %642 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %643 = tensor.splat %642 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<6x32xi32>
    %644 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%636, %641 : tensor<6x1024xi8>, tensor<1024x32xi8>) outs(%643 : tensor<6x32xi32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb80(%645: i8, %646: i8, %647: i32):
      %648 = arith.extsi %645 : i8 to i32
      %649 = arith.extsi %646 : i8 to i32
      %650 = arith.muli %648, %649 : i32
      %651 = arith.addi %647, %650 : i32
      linalg.yield %651 : i32
    } -> tensor<6x32xi32>
    %652 = tensor.concat dim(0) %644 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int32"} : (tensor<6x32xi32>) -> tensor<6x32xi32>
    %653 = tensor.collapse_shape %652 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<6x32xi32> into tensor<192xi32>
    %654 = tensor.expand_shape %653 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<192xi32> into tensor<1x6x32xi32>
    %655 = tensor.empty() : tensor<1x6x32xf32>
    %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%654 : tensor<1x6x32xi32>) outs(%655 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb81(%657: i32, %658: f32):
      %659 = arith.sitofp %657 : i32 to f32
      linalg.yield %659 : f32
    } -> tensor<1x6x32xf32>
    %660 = tensor.collapse_shape %488 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x6xf32> into tensor<6xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6xf32> into tensor<1x6x1xf32>
    %662 = tensor.empty() : tensor<1x6x32xf32>
    %663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%656, %661 : tensor<1x6x32xf32>, tensor<1x6x1xf32>) outs(%662 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb82(%664: f32, %665: f32, %666: f32):
      %667 = arith.mulf %664, %665 : f32
      linalg.yield %667 : f32
    } -> tensor<1x6x32xf32>
    %668 = tensor.collapse_shape %589 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %669 = tensor.expand_shape %668 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
    %670 = tensor.empty() : tensor<1x6x32xf32>
    %671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663, %669 : tensor<1x6x32xf32>, tensor<1x1x32xf32>) outs(%670 : tensor<1x6x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb83(%672: f32, %673: f32, %674: f32):
      %675 = arith.mulf %672, %673 : f32
      linalg.yield %675 : f32
    } -> tensor<1x6x32xf32>
    %676 = tensor.collapse_shape %671 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x6x32xf32> into tensor<192xf32>
    %677 = tensor.expand_shape %676 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 6, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<192xf32> into tensor<1x1x6x32xf32>
    func.return %677 : tensor<1x1x6x32xf32>
  }
}
