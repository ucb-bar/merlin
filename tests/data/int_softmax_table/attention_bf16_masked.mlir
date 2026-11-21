builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x2x20x32xbf16>, %1: tensor<1x2x24x32xbf16>, %2: tensor<1x2x24x32xbf16>, %3: tensor<1x1x20x24xbf16>) -> tensor<1x2x20x32xbf16> {
    %4 = tensor.empty() : tensor<1x2x32x24xbf16>
    %5 = linalg.transpose ins(%1:tensor<1x2x24x32xbf16>) outs(%4:tensor<1x2x32x24xbf16>) permutation = [0, 1, 3, 2]
    %6 = tensor.empty() : tensor<1x2x20x32xbf16>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x2x20x32xbf16>) outs(%6 : tensor<1x2x20x32xbf16>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%8: bf16, %9: bf16):
      linalg.yield %8 : bf16
    } -> tensor<1x2x20x32xbf16>
    %10 = tensor.collapse_shape %7 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1x2x20x32xbf16> into tensor<1280xbf16>
    %11 = tensor.expand_shape %10 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 32] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1280xbf16> into tensor<2x20x32xbf16>
    %12 = tensor.empty() : tensor<1x2x32x24xbf16>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<1x2x32x24xbf16>) outs(%12 : tensor<1x2x32x24xbf16>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%14: bf16, %15: bf16):
      linalg.yield %14 : bf16
    } -> tensor<1x2x32x24xbf16>
    %16 = tensor.empty() : tensor<1x2x24x32xbf16>
    %17 = linalg.transpose ins(%13:tensor<1x2x32x24xbf16>) outs(%16:tensor<1x2x24x32xbf16>) permutation = [0, 1, 3, 2]
    %18 = tensor.collapse_shape %17 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1x2x24x32xbf16> into tensor<1536xbf16>
    %19 = tensor.expand_shape %18 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 24, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1536xbf16> into tensor<2x24x32xbf16>
    %20 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} 0x7f80 : bf16
    %21 = tensor.splat %20 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %22 = linalg.reduce ins(%11:tensor<2x20x32xbf16>) outs(%21:tensor<2x20xbf16>) dimensions = [2]
    (%23: bf16, %24: bf16) {
      %25 = arith.minimumf %23, %24 : bf16
      linalg.yield %25 : bf16
    }
    %26 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %27 = tensor.splat %26 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %28 = linalg.reduce ins(%11:tensor<2x20x32xbf16>) outs(%27:tensor<2x20xbf16>) dimensions = [2]
    (%29: bf16, %30: bf16) {
      %31 = arith.maximumf %29, %30 : bf16
      linalg.yield %31 : bf16
    }
    %32 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %33 = tensor.splat %32 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %34 = tensor.empty() : tensor<2x20xbf16>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %33 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%34 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%36: bf16, %37: bf16, %38: bf16):
      %39 = arith.minimumf %36, %37 : bf16
      linalg.yield %39 : bf16
    } -> tensor<2x20xbf16>
    %40 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %41 = tensor.splat %40 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %42 = tensor.empty() : tensor<2x20xbf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %41 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%42 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%44: bf16, %45: bf16, %46: bf16):
      %47 = arith.maximumf %44, %45 : bf16
      linalg.yield %47 : bf16
    } -> tensor<2x20xbf16>
    %48 = tensor.empty() : tensor<2x20xbf16>
    %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<2x20xbf16>) outs(%48 : tensor<2x20xbf16>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "bfloat16"} {
    ^bb4(%50: bf16, %51: bf16):
      %52 = arith.negf %50 : bf16
      linalg.yield %52 : bf16
    } -> tensor<2x20xbf16>
    %53 = tensor.empty() : tensor<2x20xbf16>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%49, %43 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%53 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb5(%55: bf16, %56: bf16, %57: bf16):
      %58 = arith.maximumf %55, %56 : bf16
      linalg.yield %58 : bf16
    } -> tensor<2x20xbf16>
    %59 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 1.270000e+02 : bf16
    %60 = tensor.splat %59 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %61 = tensor.empty() : tensor<2x20xbf16>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%54, %60 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%61 : tensor<2x20xbf16>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb6(%63: bf16, %64: bf16, %65: bf16):
      %66 = arith.divf %63, %64 : bf16
      linalg.yield %66 : bf16
    } -> tensor<2x20xbf16>
    %67 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %68 = tensor.splat %67 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %69 = tensor.empty() : tensor<2x20xbf16>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62 : tensor<2x20xbf16>) outs(%69 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb7(%71: bf16, %72: bf16):
      %73 = arith.constant 1.001360e-05 : bf16
      %74 = arith.maximumf %71, %73 : bf16
      linalg.yield %74 : bf16
    } -> tensor<2x20xbf16>
    %75 = tensor.collapse_shape %70 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %76 = tensor.expand_shape %75 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %77 = tensor.collapse_shape %68 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %78 = tensor.expand_shape %77 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %79 = tensor.empty() : tensor<2x20x1xbf16>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<2x20x1xbf16>) outs(%79 : tensor<2x20x1xbf16>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "bfloat16"} {
    ^bb8(%81: bf16, %82: bf16):
      %83 = arith.constant 1.000000e+00 : bf16
      %84 = arith.divf %83, %81 : bf16
      linalg.yield %84 : bf16
    } -> tensor<2x20x1xbf16>
    %85 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.000000e+00 : bf16
    %86 = tensor.splat %85 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x20x1xbf16>
    %87 = tensor.empty() : tensor<2x20x1xbf16>
    %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %86 : tensor<2x20x1xbf16>, tensor<2x20x1xbf16>) outs(%87 : tensor<2x20x1xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb9(%89: bf16, %90: bf16, %91: bf16):
      %92 = arith.mulf %89, %90 : bf16
      linalg.yield %92 : bf16
    } -> tensor<2x20x1xbf16>
    %93 = tensor.empty() : tensor<2x20x32xbf16>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %88 : tensor<2x20x32xbf16>, tensor<2x20x1xbf16>) outs(%93 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb10(%95: bf16, %96: bf16, %97: bf16):
      %98 = arith.mulf %95, %96 : bf16
      linalg.yield %98 : bf16
    } -> tensor<2x20x32xbf16>
    %99 = tensor.empty() : tensor<2x20x32xbf16>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94 : tensor<2x20x32xbf16>) outs(%99 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "bfloat16"} {
    ^bb11(%101: bf16, %102: bf16):
      %103 = math.roundeven %101 : bf16
      linalg.yield %103 : bf16
    } -> tensor<2x20x32xbf16>
    %104 = tensor.empty() : tensor<2x20x32xbf16>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %78 : tensor<2x20x32xbf16>, tensor<2x20x1xbf16>) outs(%104 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb12(%106: bf16, %107: bf16, %108: bf16):
      %109 = arith.addf %106, %107 : bf16
      linalg.yield %109 : bf16
    } -> tensor<2x20x32xbf16>
    %110 = tensor.empty() : tensor<2x20x32xbf16>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105 : tensor<2x20x32xbf16>) outs(%110 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb13(%112: bf16, %113: bf16):
      %114 = arith.constant -1.270000e+02 : bf16
      %115 = arith.maximumf %112, %114 : bf16
      %116 = arith.constant 1.270000e+02 : bf16
      %117 = arith.minimumf %115, %116 : bf16
      linalg.yield %117 : bf16
    } -> tensor<2x20x32xbf16>
    %118 = tensor.empty() : tensor<2x20x32xi8>
    %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%111 : tensor<2x20x32xbf16>) outs(%118 : tensor<2x20x32xi8>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb14(%120: bf16, %121: i8):
      %122 = arith.fptosi %120 : bf16 to i8
      linalg.yield %122 : i8
    } -> tensor<2x20x32xi8>
    %123 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} 0x7f80 : bf16
    %124 = tensor.splat %123 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16>
    %125 = linalg.reduce ins(%19:tensor<2x24x32xbf16>) outs(%124:tensor<2x24xbf16>) dimensions = [2]
    (%126: bf16, %127: bf16) {
      %128 = arith.minimumf %126, %127 : bf16
      linalg.yield %128 : bf16
    }
    %129 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %130 = tensor.splat %129 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16>
    %131 = linalg.reduce ins(%19:tensor<2x24x32xbf16>) outs(%130:tensor<2x24xbf16>) dimensions = [2]
    (%132: bf16, %133: bf16) {
      %134 = arith.maximumf %132, %133 : bf16
      linalg.yield %134 : bf16
    }
    %135 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %136 = tensor.splat %135 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16>
    %137 = tensor.empty() : tensor<2x24xbf16>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %136 : tensor<2x24xbf16>, tensor<2x24xbf16>) outs(%137 : tensor<2x24xbf16>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "bfloat16"} {
    ^bb15(%139: bf16, %140: bf16, %141: bf16):
      %142 = arith.minimumf %139, %140 : bf16
      linalg.yield %142 : bf16
    } -> tensor<2x24xbf16>
    %143 = arith.constant {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %144 = tensor.splat %143 {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16>
    %145 = tensor.empty() : tensor<2x24xbf16>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%131, %144 : tensor<2x24xbf16>, tensor<2x24xbf16>) outs(%145 : tensor<2x24xbf16>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb16(%147: bf16, %148: bf16, %149: bf16):
      %150 = arith.maximumf %147, %148 : bf16
      linalg.yield %150 : bf16
    } -> tensor<2x24xbf16>
    %151 = tensor.empty() : tensor<2x24xbf16>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%138 : tensor<2x24xbf16>) outs(%151 : tensor<2x24xbf16>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "bfloat16"} {
    ^bb17(%153: bf16, %154: bf16):
      %155 = arith.negf %153 : bf16
      linalg.yield %155 : bf16
    } -> tensor<2x24xbf16>
    %156 = tensor.empty() : tensor<2x24xbf16>
    %157 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%152, %146 : tensor<2x24xbf16>, tensor<2x24xbf16>) outs(%156 : tensor<2x24xbf16>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb18(%158: bf16, %159: bf16, %160: bf16):
      %161 = arith.maximumf %158, %159 : bf16
      linalg.yield %161 : bf16
    } -> tensor<2x24xbf16>
    %162 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 1.275000e+02 : bf16
    %163 = tensor.splat %162 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16>
    %164 = tensor.empty() : tensor<2x24xbf16>
    %165 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%157, %163 : tensor<2x24xbf16>, tensor<2x24xbf16>) outs(%164 : tensor<2x24xbf16>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb19(%166: bf16, %167: bf16, %168: bf16):
      %169 = arith.divf %166, %167 : bf16
      linalg.yield %169 : bf16
    } -> tensor<2x24xbf16>
    %170 = tensor.empty() : tensor<2x24xbf16>
    %171 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%165 : tensor<2x24xbf16>) outs(%170 : tensor<2x24xbf16>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb20(%172: bf16, %173: bf16):
      %174 = arith.constant 1.192090e-07 : bf16
      %175 = arith.maximumf %172, %174 : bf16
      linalg.yield %175 : bf16
    } -> tensor<2x24xbf16>
    %176 = tensor.collapse_shape %171 [[0 : i64, 1 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16> into tensor<48xbf16>
    %177 = tensor.expand_shape %176 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 24, 1] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<48xbf16> into tensor<2x24x1xbf16>
    %178 = tensor.empty() : tensor<2x24x1xbf16>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%177 : tensor<2x24x1xbf16>) outs(%178 : tensor<2x24x1xbf16>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "bfloat16"} {
    ^bb21(%180: bf16, %181: bf16):
      %182 = arith.constant 1.000000e+00 : bf16
      %183 = arith.divf %182, %180 : bf16
      linalg.yield %183 : bf16
    } -> tensor<2x24x1xbf16>
    %184 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.000000e+00 : bf16
    %185 = tensor.splat %184 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x24x1xbf16>
    %186 = tensor.empty() : tensor<2x24x1xbf16>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%179, %185 : tensor<2x24x1xbf16>, tensor<2x24x1xbf16>) outs(%186 : tensor<2x24x1xbf16>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb22(%188: bf16, %189: bf16, %190: bf16):
      %191 = arith.mulf %188, %189 : bf16
      linalg.yield %191 : bf16
    } -> tensor<2x24x1xbf16>
    %192 = tensor.empty() : tensor<2x24x32xbf16>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %187 : tensor<2x24x32xbf16>, tensor<2x24x1xbf16>) outs(%192 : tensor<2x24x32xbf16>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb23(%194: bf16, %195: bf16, %196: bf16):
      %197 = arith.mulf %194, %195 : bf16
      linalg.yield %197 : bf16
    } -> tensor<2x24x32xbf16>
    %198 = tensor.empty() : tensor<2x24x32xbf16>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%193 : tensor<2x24x32xbf16>) outs(%198 : tensor<2x24x32xbf16>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "bfloat16"} {
    ^bb24(%200: bf16, %201: bf16):
      %202 = math.roundeven %200 : bf16
      linalg.yield %202 : bf16
    } -> tensor<2x24x32xbf16>
    %203 = tensor.empty() : tensor<2x24x32xbf16>
    %204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%199 : tensor<2x24x32xbf16>) outs(%203 : tensor<2x24x32xbf16>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb25(%205: bf16, %206: bf16):
      %207 = arith.constant -1.280000e+02 : bf16
      %208 = arith.maximumf %205, %207 : bf16
      %209 = arith.constant 1.270000e+02 : bf16
      %210 = arith.minimumf %208, %209 : bf16
      linalg.yield %210 : bf16
    } -> tensor<2x24x32xbf16>
    %211 = tensor.empty() : tensor<2x24x32xi8>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%204 : tensor<2x24x32xbf16>) outs(%211 : tensor<2x24x32xi8>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb26(%213: bf16, %214: i8):
      %215 = arith.fptosi %213 : bf16 to i8
      linalg.yield %215 : i8
    } -> tensor<2x24x32xi8>
    %216 = "tensor.extract_slice"(%119) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 20, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x20x32xi8>) -> tensor<1x20x32xi8>
    %217 = tensor.collapse_shape %216 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x20x32xi8> into tensor<640xi8>
    %218 = tensor.expand_shape %217 [[0 : i64, 1 : i64]] output_shape [20, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<640xi8> into tensor<20x32xi8>
    %219 = "tensor.extract_slice"(%212) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 24, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x24x32xi8>) -> tensor<1x24x32xi8>
    %220 = tensor.collapse_shape %219 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x24x32xi8> into tensor<768xi8>
    %221 = tensor.expand_shape %220 [[0 : i64, 1 : i64]] output_shape [24, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<768xi8> into tensor<24x32xi8>
    %222 = tensor.empty() : tensor<32x24xi8>
    %223 = linalg.transpose ins(%221:tensor<24x32xi8>) outs(%222:tensor<32x24xi8>) permutation = [1, 0]
    %224 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %225 = tensor.splat %224 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<20x24xi32>
    %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%218, %223 : tensor<20x32xi8>, tensor<32x24xi8>) outs(%225 : tensor<20x24xi32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb27(%227: i8, %228: i8, %229: i32):
      %230 = arith.extsi %227 : i8 to i32
      %231 = arith.extsi %228 : i8 to i32
      %232 = arith.muli %230, %231 : i32
      %233 = arith.addi %229, %232 : i32
      linalg.yield %233 : i32
    } -> tensor<20x24xi32>
    %234 = "tensor.extract_slice"(%119) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 20, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x20x32xi8>) -> tensor<1x20x32xi8>
    %235 = tensor.collapse_shape %234 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x20x32xi8> into tensor<640xi8>
    %236 = tensor.expand_shape %235 [[0 : i64, 1 : i64]] output_shape [20, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<640xi8> into tensor<20x32xi8>
    %237 = "tensor.extract_slice"(%212) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 24, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x24x32xi8>) -> tensor<1x24x32xi8>
    %238 = tensor.collapse_shape %237 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x24x32xi8> into tensor<768xi8>
    %239 = tensor.expand_shape %238 [[0 : i64, 1 : i64]] output_shape [24, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<768xi8> into tensor<24x32xi8>
    %240 = tensor.empty() : tensor<32x24xi8>
    %241 = linalg.transpose ins(%239:tensor<24x32xi8>) outs(%240:tensor<32x24xi8>) permutation = [1, 0]
    %242 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %243 = tensor.splat %242 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<20x24xi32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%236, %241 : tensor<20x32xi8>, tensor<32x24xi8>) outs(%243 : tensor<20x24xi32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb28(%245: i8, %246: i8, %247: i32):
      %248 = arith.extsi %245 : i8 to i32
      %249 = arith.extsi %246 : i8 to i32
      %250 = arith.muli %248, %249 : i32
      %251 = arith.addi %247, %250 : i32
      linalg.yield %251 : i32
    } -> tensor<20x24xi32>
    %252 = tensor.concat dim(0) %226, %244 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int32"} : (tensor<20x24xi32>, tensor<20x24xi32>) -> tensor<40x24xi32>
    %253 = tensor.collapse_shape %252 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<40x24xi32> into tensor<960xi32>
    %254 = tensor.expand_shape %253 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 24] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<960xi32> into tensor<2x20x24xi32>
    %255 = tensor.empty() : tensor<2x20x24xbf16>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254 : tensor<2x20x24xi32>) outs(%255 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb29(%257: i32, %258: bf16):
      %259 = arith.sitofp %257 : i32 to bf16
      linalg.yield %259 : bf16
    } -> tensor<2x20x24xbf16>
    %260 = tensor.collapse_shape %70 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %261 = tensor.expand_shape %260 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %262 = tensor.empty() : tensor<2x20x24xbf16>
    %263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%256, %261 : tensor<2x20x24xbf16>, tensor<2x20x1xbf16>) outs(%262 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb30(%264: bf16, %265: bf16, %266: bf16):
      %267 = arith.mulf %264, %265 : bf16
      linalg.yield %267 : bf16
    } -> tensor<2x20x24xbf16>
    %268 = tensor.collapse_shape %171 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<2x24xbf16> into tensor<48xbf16>
    %269 = tensor.expand_shape %268 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 1, 24] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<48xbf16> into tensor<2x1x24xbf16>
    %270 = tensor.empty() : tensor<2x20x24xbf16>
    %271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%263, %269 : tensor<2x20x24xbf16>, tensor<2x1x24xbf16>) outs(%270 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb31(%272: bf16, %273: bf16, %274: bf16):
      %275 = arith.mulf %272, %273 : bf16
      linalg.yield %275 : bf16
    } -> tensor<2x20x24xbf16>
    %276 = tensor.collapse_shape %271 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20x24xbf16> into tensor<960xbf16>
    %277 = tensor.expand_shape %276 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 20, 24] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<960xbf16> into tensor<1x2x20x24xbf16>
    %278 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.767580e-01 : bf16
    %279 = tensor.splat %278 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<1x2x20x24xbf16>
    %280 = tensor.empty() : tensor<1x2x20x24xbf16>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%277, %279 : tensor<1x2x20x24xbf16>, tensor<1x2x20x24xbf16>) outs(%280 : tensor<1x2x20x24xbf16>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb32(%282: bf16, %283: bf16, %284: bf16):
      %285 = arith.mulf %282, %283 : bf16
      linalg.yield %285 : bf16
    } -> tensor<1x2x20x24xbf16>
    %286 = tensor.empty() : tensor<1x2x20x24xbf16>
    %287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%281, %3 : tensor<1x2x20x24xbf16>, tensor<1x1x20x24xbf16>) outs(%286 : tensor<1x2x20x24xbf16>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb33(%288: bf16, %289: bf16, %290: bf16):
      %291 = arith.addf %288, %289 : bf16
      linalg.yield %291 : bf16
    } -> tensor<1x2x20x24xbf16>
    %292 = tensor.empty() : tensor<1x2x20x24xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%287 : tensor<1x2x20x24xbf16>) outs(%292 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb34(%294: bf16, %295: f32):
      %296 = arith.extf %294 : bf16 to f32
      linalg.yield %296 : f32
    } -> tensor<1x2x20x24xf32>
    %297 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %298 = tensor.splat %297 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x2x20xf32>
    %299 = linalg.reduce ins(%293:tensor<1x2x20x24xf32>) outs(%298:tensor<1x2x20xf32>) dimensions = [3]
    (%300: f32, %301: f32) {
      %302 = arith.maximumf %300, %301 : f32
      linalg.yield %302 : f32
    }
    %303 = tensor.collapse_shape %299 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x2x20xf32> into tensor<40xf32>
    %304 = tensor.expand_shape %303 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 20, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<40xf32> into tensor<1x2x20x1xf32>
    %305 = tensor.empty() : tensor<1x2x20x24xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%293, %304 : tensor<1x2x20x24xf32>, tensor<1x2x20x1xf32>) outs(%305 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb35(%307: f32, %308: f32, %309: f32):
      %310 = arith.subf %307, %308 : f32
      linalg.yield %310 : f32
    } -> tensor<1x2x20x24xf32>
    %311 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 0.00270760618 : f32
    %312 = tensor.splat %311 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x2x20x24xf32>
    %313 = tensor.empty() : tensor<1x2x20x24xf32>
    %314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%306, %312 : tensor<1x2x20x24xf32>, tensor<1x2x20x24xf32>) outs(%313 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb36(%315: f32, %316: f32, %317: f32):
      %318 = arith.divf %315, %316 : f32
      linalg.yield %318 : f32
    } -> tensor<1x2x20x24xf32>
    %319 = tensor.empty() : tensor<1x2x20x24xf32>
    %320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%314 : tensor<1x2x20x24xf32>) outs(%319 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb37(%321: f32, %322: f32):
      %323 = math.roundeven %321 : f32
      linalg.yield %323 : f32
    } -> tensor<1x2x20x24xf32>
    %324 = tensor.empty() : tensor<1x2x20x24xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%320 : tensor<1x2x20x24xf32>) outs(%324 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb38(%326: f32, %327: f32):
      %328 = arith.constant -1.108000e+04 : f32
      %329 = arith.maximumf %326, %328 : f32
      %330 = arith.constant 0.000000e+00 : f32
      %331 = arith.minimumf %329, %330 : f32
      linalg.yield %331 : f32
    } -> tensor<1x2x20x24xf32>
    %332 = tensor.empty() : tensor<1x2x20x24xi32>
    %333 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%325 : tensor<1x2x20x24xf32>) outs(%332 : tensor<1x2x20x24xi32>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb39(%334: f32, %335: i32):
      %336 = arith.fptosi %334 : f32 to i32
      linalg.yield %336 : i32
    } -> tensor<1x2x20x24xi32>
    %337 = tensor.empty() : tensor<1x2x20x24xi64>
    %338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333 : tensor<1x2x20x24xi32>) outs(%337 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int64"} {
    ^bb40(%339: i32, %340: i64):
      %341 = arith.extsi %339 : i32 to i64
      linalg.yield %341 : i64
    } -> tensor<1x2x20x24xi64>
    %342 = arith.constant {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} 0 : i64
    %343 = tensor.splat %342 {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %344 = tensor.empty() : tensor<1x2x20x24xi64>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %338 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%344 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} {
    ^bb41(%346: i64, %347: i64, %348: i64):
      %349 = arith.subi %346, %347 : i64
      linalg.yield %349 : i64
    } -> tensor<1x2x20x24xi64>
    %350 = tensor.empty() : tensor<1x2x20x24xi64>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%345 : tensor<1x2x20x24xi64>) outs(%350 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor_Scalar", prov.orig_dtype = "int64"} {
    ^bb42(%352: i64, %353: i64):
      %354 = arith.constant 8 : i64
      %355 = arith.shrsi %352, %354 : i64
      linalg.yield %355 : i64
    } -> tensor<1x2x20x24xi64>
    %356 = arith.constant {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 256 : i64
    %357 = tensor.splat %356 {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %358 = tensor.empty() : tensor<1x2x20x24xi64>
    %359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%351, %357 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%358 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb43(%360: i64, %361: i64, %362: i64):
      %363 = arith.muli %360, %361 : i64
      linalg.yield %363 : i64
    } -> tensor<1x2x20x24xi64>
    %364 = tensor.empty() : tensor<1x2x20x24xi64>
    %365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%338, %359 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%364 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb44(%366: i64, %367: i64, %368: i64):
      %369 = arith.addi %366, %367 : i64
      linalg.yield %369 : i64
    } -> tensor<1x2x20x24xi64>
    %370 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 500 : i64
    %371 = tensor.splat %370 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %372 = tensor.empty() : tensor<1x2x20x24xi64>
    %373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%365, %371 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%372 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb45(%374: i64, %375: i64, %376: i64):
      %377 = arith.addi %374, %375 : i64
      linalg.yield %377 : i64
    } -> tensor<1x2x20x24xi64>
    %378 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 2819 : i64
    %379 = tensor.splat %378 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %380 = tensor.empty() : tensor<1x2x20x24xi64>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%373, %379 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%380 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb46(%382: i64, %383: i64, %384: i64):
      %385 = arith.muli %382, %383 : i64
      linalg.yield %385 : i64
    } -> tensor<1x2x20x24xi64>
    %386 = tensor.empty() : tensor<1x2x20x24xi64>
    %387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %373 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%386 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb47(%388: i64, %389: i64, %390: i64):
      %391 = arith.muli %388, %389 : i64
      linalg.yield %391 : i64
    } -> tensor<1x2x20x24xi64>
    %392 = arith.constant {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 369388222 : i64
    %393 = tensor.splat %392 {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %394 = tensor.empty() : tensor<1x2x20x24xi64>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387, %393 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%394 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb48(%396: i64, %397: i64, %398: i64):
      %399 = arith.addi %396, %397 : i64
      linalg.yield %399 : i64
    } -> tensor<1x2x20x24xi64>
    %400 = tensor.empty() : tensor<1x2x20x24xi64>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %351 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%400 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor", prov.orig_dtype = "int64"} {
    ^bb49(%402: i64, %403: i64, %404: i64):
      %405 = arith.constant 63 : i64
      %406 = arith.minui %403, %405 : i64
      %407 = arith.shrsi %402, %406 : i64
      linalg.yield %407 : i64
    } -> tensor<1x2x20x24xi64>
    %408 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 127 : i64
    %409 = tensor.splat %408 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %410 = tensor.empty() : tensor<1x2x20x24xi64>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401, %409 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%410 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb50(%412: i64, %413: i64, %414: i64):
      %415 = arith.muli %412, %413 : i64
      linalg.yield %415 : i64
    } -> tensor<1x2x20x24xi64>
    %416 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 537069111 : i64
    %417 = tensor.splat %416 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<1x2x20x24xi64>
    %418 = tensor.empty() : tensor<1x2x20x24xi64>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411, %417 : tensor<1x2x20x24xi64>, tensor<1x2x20x24xi64>) outs(%418 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb51(%420: i64, %421: i64, %422: i64):
      %423 = arith.addi %420, %421 : i64
      linalg.yield %423 : i64
    } -> tensor<1x2x20x24xi64>
    %424 = tensor.empty() : tensor<1x2x20x24xi64>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419 : tensor<1x2x20x24xi64>) outs(%424 : tensor<1x2x20x24xi64>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "floor_divide", prov.op = "floor_divide", prov.aten = "aten.div.Tensor_mode", prov.orig_dtype = "int64"} {
    ^bb52(%426: i64, %427: i64):
      %428 = arith.constant 1074138222 : i64
      %429 = arith.floordivsi %426, %428 : i64
      linalg.yield %429 : i64
    } -> tensor<1x2x20x24xi64>
    %430 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} 0 : i64
    %431 = tensor.splat %430 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<1x2x20xi64>
    %432 = linalg.reduce ins(%425:tensor<1x2x20x24xi64>) outs(%431:tensor<1x2x20xi64>) dimensions = [3]
    (%433: i64, %434: i64) {
      %435 = arith.addi %433, %434 : i64
      linalg.yield %435 : i64
    }
    %436 = tensor.collapse_shape %432 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<1x2x20xi64> into tensor<40xi64>
    %437 = tensor.expand_shape %436 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 20, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<40xi64> into tensor<1x2x20x1xi64>
    %438 = tensor.empty() : tensor<1x2x20x24xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%425 : tensor<1x2x20x24xi64>) outs(%438 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb53(%440: i64, %441: f32):
      %442 = arith.sitofp %440 : i64 to f32
      linalg.yield %442 : f32
    } -> tensor<1x2x20x24xf32>
    %443 = tensor.empty() : tensor<1x2x20x1xf32>
    %444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%437 : tensor<1x2x20x1xi64>) outs(%443 : tensor<1x2x20x1xf32>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb54(%445: i64, %446: f32):
      %447 = arith.sitofp %445 : i64 to f32
      linalg.yield %447 : f32
    } -> tensor<1x2x20x1xf32>
    %448 = tensor.empty() : tensor<1x2x20x24xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %444 : tensor<1x2x20x24xf32>, tensor<1x2x20x1xf32>) outs(%448 : tensor<1x2x20x24xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%450: f32, %451: f32, %452: f32):
      %453 = arith.divf %450, %451 : f32
      linalg.yield %453 : f32
    } -> tensor<1x2x20x24xf32>
    %454 = tensor.empty() : tensor<1x2x20x24xbf16>
    %455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%449 : tensor<1x2x20x24xf32>) outs(%454 : tensor<1x2x20x24xbf16>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb56(%456: f32, %457: bf16):
      %458 = arith.truncf %456 : f32 to bf16
      linalg.yield %458 : bf16
    } -> tensor<1x2x20x24xbf16>
    %459 = tensor.empty() : tensor<1x2x20x24xbf16>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%455 : tensor<1x2x20x24xbf16>) outs(%459 : tensor<1x2x20x24xbf16>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bfloat16"} {
    ^bb57(%461: bf16, %462: bf16):
      linalg.yield %461 : bf16
    } -> tensor<1x2x20x24xbf16>
    %463 = tensor.collapse_shape %460 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1x2x20x24xbf16> into tensor<960xbf16>
    %464 = tensor.expand_shape %463 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 24] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<960xbf16> into tensor<2x20x24xbf16>
    %465 = tensor.empty() : tensor<1x2x24x32xbf16>
    %466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x2x24x32xbf16>) outs(%465 : tensor<1x2x24x32xbf16>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bfloat16"} {
    ^bb58(%467: bf16, %468: bf16):
      linalg.yield %467 : bf16
    } -> tensor<1x2x24x32xbf16>
    %469 = tensor.empty() : tensor<1x2x32x24xbf16>
    %470 = linalg.transpose ins(%466:tensor<1x2x24x32xbf16>) outs(%469:tensor<1x2x32x24xbf16>) permutation = [0, 1, 3, 2]
    %471 = tensor.collapse_shape %470 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1x2x32x24xbf16> into tensor<1536xbf16>
    %472 = tensor.expand_shape %471 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 24] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1536xbf16> into tensor<2x32x24xbf16>
    %473 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} 0x7f80 : bf16
    %474 = tensor.splat %473 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %475 = linalg.reduce ins(%464:tensor<2x20x24xbf16>) outs(%474:tensor<2x20xbf16>) dimensions = [2]
    (%476: bf16, %477: bf16) {
      %478 = arith.minimumf %476, %477 : bf16
      linalg.yield %478 : bf16
    }
    %479 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %480 = tensor.splat %479 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %481 = linalg.reduce ins(%464:tensor<2x20x24xbf16>) outs(%480:tensor<2x20xbf16>) dimensions = [2]
    (%482: bf16, %483: bf16) {
      %484 = arith.maximumf %482, %483 : bf16
      linalg.yield %484 : bf16
    }
    %485 = arith.constant {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %486 = tensor.splat %485 {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %487 = tensor.empty() : tensor<2x20xbf16>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%475, %486 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%487 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "bfloat16"} {
    ^bb59(%489: bf16, %490: bf16, %491: bf16):
      %492 = arith.minimumf %489, %490 : bf16
      linalg.yield %492 : bf16
    } -> tensor<2x20xbf16>
    %493 = arith.constant {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %494 = tensor.splat %493 {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %495 = tensor.empty() : tensor<2x20xbf16>
    %496 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%481, %494 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%495 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb60(%497: bf16, %498: bf16, %499: bf16):
      %500 = arith.maximumf %497, %498 : bf16
      linalg.yield %500 : bf16
    } -> tensor<2x20xbf16>
    %501 = tensor.empty() : tensor<2x20xbf16>
    %502 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%488 : tensor<2x20xbf16>) outs(%501 : tensor<2x20xbf16>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "bfloat16"} {
    ^bb61(%503: bf16, %504: bf16):
      %505 = arith.negf %503 : bf16
      linalg.yield %505 : bf16
    } -> tensor<2x20xbf16>
    %506 = tensor.empty() : tensor<2x20xbf16>
    %507 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%502, %496 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%506 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb62(%508: bf16, %509: bf16, %510: bf16):
      %511 = arith.maximumf %508, %509 : bf16
      linalg.yield %511 : bf16
    } -> tensor<2x20xbf16>
    %512 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 1.270000e+02 : bf16
    %513 = tensor.splat %512 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %514 = tensor.empty() : tensor<2x20xbf16>
    %515 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%507, %513 : tensor<2x20xbf16>, tensor<2x20xbf16>) outs(%514 : tensor<2x20xbf16>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb63(%516: bf16, %517: bf16, %518: bf16):
      %519 = arith.divf %516, %517 : bf16
      linalg.yield %519 : bf16
    } -> tensor<2x20xbf16>
    %520 = arith.constant {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %521 = tensor.splat %520 {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16>
    %522 = tensor.empty() : tensor<2x20xbf16>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%515 : tensor<2x20xbf16>) outs(%522 : tensor<2x20xbf16>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb64(%524: bf16, %525: bf16):
      %526 = arith.constant 1.001360e-05 : bf16
      %527 = arith.maximumf %524, %526 : bf16
      linalg.yield %527 : bf16
    } -> tensor<2x20xbf16>
    %528 = tensor.collapse_shape %523 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %529 = tensor.expand_shape %528 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %530 = tensor.collapse_shape %521 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %531 = tensor.expand_shape %530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %532 = tensor.empty() : tensor<2x20x1xbf16>
    %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%529 : tensor<2x20x1xbf16>) outs(%532 : tensor<2x20x1xbf16>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "bfloat16"} {
    ^bb65(%534: bf16, %535: bf16):
      %536 = arith.constant 1.000000e+00 : bf16
      %537 = arith.divf %536, %534 : bf16
      linalg.yield %537 : bf16
    } -> tensor<2x20x1xbf16>
    %538 = arith.constant {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.000000e+00 : bf16
    %539 = tensor.splat %538 {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x20x1xbf16>
    %540 = tensor.empty() : tensor<2x20x1xbf16>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%533, %539 : tensor<2x20x1xbf16>, tensor<2x20x1xbf16>) outs(%540 : tensor<2x20x1xbf16>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb66(%542: bf16, %543: bf16, %544: bf16):
      %545 = arith.mulf %542, %543 : bf16
      linalg.yield %545 : bf16
    } -> tensor<2x20x1xbf16>
    %546 = tensor.empty() : tensor<2x20x24xbf16>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%464, %541 : tensor<2x20x24xbf16>, tensor<2x20x1xbf16>) outs(%546 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb67(%548: bf16, %549: bf16, %550: bf16):
      %551 = arith.mulf %548, %549 : bf16
      linalg.yield %551 : bf16
    } -> tensor<2x20x24xbf16>
    %552 = tensor.empty() : tensor<2x20x24xbf16>
    %553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%547 : tensor<2x20x24xbf16>) outs(%552 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "bfloat16"} {
    ^bb68(%554: bf16, %555: bf16):
      %556 = math.roundeven %554 : bf16
      linalg.yield %556 : bf16
    } -> tensor<2x20x24xbf16>
    %557 = tensor.empty() : tensor<2x20x24xbf16>
    %558 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553, %531 : tensor<2x20x24xbf16>, tensor<2x20x1xbf16>) outs(%557 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb69(%559: bf16, %560: bf16, %561: bf16):
      %562 = arith.addf %559, %560 : bf16
      linalg.yield %562 : bf16
    } -> tensor<2x20x24xbf16>
    %563 = tensor.empty() : tensor<2x20x24xbf16>
    %564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%558 : tensor<2x20x24xbf16>) outs(%563 : tensor<2x20x24xbf16>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb70(%565: bf16, %566: bf16):
      %567 = arith.constant -1.270000e+02 : bf16
      %568 = arith.maximumf %565, %567 : bf16
      %569 = arith.constant 1.270000e+02 : bf16
      %570 = arith.minimumf %568, %569 : bf16
      linalg.yield %570 : bf16
    } -> tensor<2x20x24xbf16>
    %571 = tensor.empty() : tensor<2x20x24xi8>
    %572 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%564 : tensor<2x20x24xbf16>) outs(%571 : tensor<2x20x24xi8>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb71(%573: bf16, %574: i8):
      %575 = arith.fptosi %573 : bf16 to i8
      linalg.yield %575 : i8
    } -> tensor<2x20x24xi8>
    %576 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} 0x7f80 : bf16
    %577 = tensor.splat %576 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amin.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16>
    %578 = linalg.reduce ins(%472:tensor<2x32x24xbf16>) outs(%577:tensor<2x32xbf16>) dimensions = [2]
    (%579: bf16, %580: bf16) {
      %581 = arith.minimumf %579, %580 : bf16
      linalg.yield %581 : bf16
    }
    %582 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %583 = tensor.splat %582 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16>
    %584 = linalg.reduce ins(%472:tensor<2x32x24xbf16>) outs(%583:tensor<2x32xbf16>) dimensions = [2]
    (%585: bf16, %586: bf16) {
      %587 = arith.maximumf %585, %586 : bf16
      linalg.yield %587 : bf16
    }
    %588 = arith.constant {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %589 = tensor.splat %588 {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16>
    %590 = tensor.empty() : tensor<2x32xbf16>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%578, %589 : tensor<2x32xbf16>, tensor<2x32xbf16>) outs(%590 : tensor<2x32xbf16>) attrs =  {prov.region_id = "minmax_16", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.minimum.default", prov.orig_dtype = "bfloat16"} {
    ^bb72(%592: bf16, %593: bf16, %594: bf16):
      %595 = arith.minimumf %592, %593 : bf16
      linalg.yield %595 : bf16
    } -> tensor<2x32xbf16>
    %596 = arith.constant {prov.region_id = "fill_9", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %597 = tensor.splat %596 {prov.region_id = "fill_9", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full_like.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16>
    %598 = tensor.empty() : tensor<2x32xbf16>
    %599 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%584, %597 : tensor<2x32xbf16>, tensor<2x32xbf16>) outs(%598 : tensor<2x32xbf16>) attrs =  {prov.region_id = "minmax_17", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb73(%600: bf16, %601: bf16, %602: bf16):
      %603 = arith.maximumf %600, %601 : bf16
      linalg.yield %603 : bf16
    } -> tensor<2x32xbf16>
    %604 = tensor.empty() : tensor<2x32xbf16>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%591 : tensor<2x32xbf16>) outs(%604 : tensor<2x32xbf16>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "bfloat16"} {
    ^bb74(%606: bf16, %607: bf16):
      %608 = arith.negf %606 : bf16
      linalg.yield %608 : bf16
    } -> tensor<2x32xbf16>
    %609 = tensor.empty() : tensor<2x32xbf16>
    %610 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%605, %599 : tensor<2x32xbf16>, tensor<2x32xbf16>) outs(%609 : tensor<2x32xbf16>) attrs =  {prov.region_id = "minmax_18", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.maximum.default", prov.orig_dtype = "bfloat16"} {
    ^bb75(%611: bf16, %612: bf16, %613: bf16):
      %614 = arith.maximumf %611, %612 : bf16
      linalg.yield %614 : bf16
    } -> tensor<2x32xbf16>
    %615 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 1.275000e+02 : bf16
    %616 = tensor.splat %615 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16>
    %617 = tensor.empty() : tensor<2x32xbf16>
    %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%610, %616 : tensor<2x32xbf16>, tensor<2x32xbf16>) outs(%617 : tensor<2x32xbf16>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb76(%619: bf16, %620: bf16, %621: bf16):
      %622 = arith.divf %619, %620 : bf16
      linalg.yield %622 : bf16
    } -> tensor<2x32xbf16>
    %623 = tensor.empty() : tensor<2x32xbf16>
    %624 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%618 : tensor<2x32xbf16>) outs(%623 : tensor<2x32xbf16>) attrs =  {prov.region_id = "minmax_19", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb77(%625: bf16, %626: bf16):
      %627 = arith.constant 1.192090e-07 : bf16
      %628 = arith.maximumf %625, %627 : bf16
      linalg.yield %628 : bf16
    } -> tensor<2x32xbf16>
    %629 = tensor.collapse_shape %624 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16> into tensor<64xbf16>
    %630 = tensor.expand_shape %629 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 32, 1] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<64xbf16> into tensor<2x32x1xbf16>
    %631 = tensor.empty() : tensor<2x32x1xbf16>
    %632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%630 : tensor<2x32x1xbf16>) outs(%631 : tensor<2x32x1xbf16>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "bfloat16"} {
    ^bb78(%633: bf16, %634: bf16):
      %635 = arith.constant 1.000000e+00 : bf16
      %636 = arith.divf %635, %633 : bf16
      linalg.yield %636 : bf16
    } -> tensor<2x32x1xbf16>
    %637 = arith.constant {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} 1.000000e+00 : bf16
    %638 = tensor.splat %637 {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} : tensor<2x32x1xbf16>
    %639 = tensor.empty() : tensor<2x32x1xbf16>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%632, %638 : tensor<2x32x1xbf16>, tensor<2x32x1xbf16>) outs(%639 : tensor<2x32x1xbf16>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb79(%641: bf16, %642: bf16, %643: bf16):
      %644 = arith.mulf %641, %642 : bf16
      linalg.yield %644 : bf16
    } -> tensor<2x32x1xbf16>
    %645 = tensor.empty() : tensor<2x32x24xbf16>
    %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%472, %640 : tensor<2x32x24xbf16>, tensor<2x32x1xbf16>) outs(%645 : tensor<2x32x24xbf16>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb80(%647: bf16, %648: bf16, %649: bf16):
      %650 = arith.mulf %647, %648 : bf16
      linalg.yield %650 : bf16
    } -> tensor<2x32x24xbf16>
    %651 = tensor.empty() : tensor<2x32x24xbf16>
    %652 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646 : tensor<2x32x24xbf16>) outs(%651 : tensor<2x32x24xbf16>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "bfloat16"} {
    ^bb81(%653: bf16, %654: bf16):
      %655 = math.roundeven %653 : bf16
      linalg.yield %655 : bf16
    } -> tensor<2x32x24xbf16>
    %656 = tensor.empty() : tensor<2x32x24xbf16>
    %657 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%652 : tensor<2x32x24xbf16>) outs(%656 : tensor<2x32x24xbf16>) attrs =  {prov.region_id = "minmax_20", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "bfloat16"} {
    ^bb82(%658: bf16, %659: bf16):
      %660 = arith.constant -1.280000e+02 : bf16
      %661 = arith.maximumf %658, %660 : bf16
      %662 = arith.constant 1.270000e+02 : bf16
      %663 = arith.minimumf %661, %662 : bf16
      linalg.yield %663 : bf16
    } -> tensor<2x32x24xbf16>
    %664 = tensor.empty() : tensor<2x32x24xi8>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%657 : tensor<2x32x24xbf16>) outs(%664 : tensor<2x32x24xi8>) attrs =  {prov.region_id = "dtype_cast_10", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb83(%666: bf16, %667: i8):
      %668 = arith.fptosi %666 : bf16 to i8
      linalg.yield %668 : i8
    } -> tensor<2x32x24xi8>
    %669 = "tensor.extract_slice"(%572) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 20, 24>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x20x24xi8>) -> tensor<1x20x24xi8>
    %670 = tensor.collapse_shape %669 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x20x24xi8> into tensor<480xi8>
    %671 = tensor.expand_shape %670 [[0 : i64, 1 : i64]] output_shape [20, 24] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<480xi8> into tensor<20x24xi8>
    %672 = "tensor.extract_slice"(%665) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 24>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x32x24xi8>) -> tensor<1x32x24xi8>
    %673 = tensor.collapse_shape %672 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x32x24xi8> into tensor<768xi8>
    %674 = tensor.expand_shape %673 [[0 : i64, 1 : i64]] output_shape [32, 24] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<768xi8> into tensor<32x24xi8>
    %675 = tensor.empty() : tensor<24x32xi8>
    %676 = linalg.transpose ins(%674:tensor<32x24xi8>) outs(%675:tensor<24x32xi8>) permutation = [1, 0]
    %677 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %678 = tensor.splat %677 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<20x32xi32>
    %679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%671, %676 : tensor<20x24xi8>, tensor<24x32xi8>) outs(%678 : tensor<20x32xi32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb84(%680: i8, %681: i8, %682: i32):
      %683 = arith.extsi %680 : i8 to i32
      %684 = arith.extsi %681 : i8 to i32
      %685 = arith.muli %683, %684 : i32
      %686 = arith.addi %682, %685 : i32
      linalg.yield %686 : i32
    } -> tensor<20x32xi32>
    %687 = "tensor.extract_slice"(%572) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 20, 24>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x20x24xi8>) -> tensor<1x20x24xi8>
    %688 = tensor.collapse_shape %687 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x20x24xi8> into tensor<480xi8>
    %689 = tensor.expand_shape %688 [[0 : i64, 1 : i64]] output_shape [20, 24] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<480xi8> into tensor<20x24xi8>
    %690 = "tensor.extract_slice"(%665) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 24>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : (tensor<2x32x24xi8>) -> tensor<1x32x24xi8>
    %691 = tensor.collapse_shape %690 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<1x32x24xi8> into tensor<768xi8>
    %692 = tensor.expand_shape %691 [[0 : i64, 1 : i64]] output_shape [32, 24] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "int8"} : tensor<768xi8> into tensor<32x24xi8>
    %693 = tensor.empty() : tensor<24x32xi8>
    %694 = linalg.transpose ins(%692:tensor<32x24xi8>) outs(%693:tensor<24x32xi8>) permutation = [1, 0]
    %695 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %696 = tensor.splat %695 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<20x32xi32>
    %697 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%689, %694 : tensor<20x24xi8>, tensor<24x32xi8>) outs(%696 : tensor<20x32xi32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb85(%698: i8, %699: i8, %700: i32):
      %701 = arith.extsi %698 : i8 to i32
      %702 = arith.extsi %699 : i8 to i32
      %703 = arith.muli %701, %702 : i32
      %704 = arith.addi %700, %703 : i32
      linalg.yield %704 : i32
    } -> tensor<20x32xi32>
    %705 = tensor.concat dim(0) %679, %697 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int32"} : (tensor<20x32xi32>, tensor<20x32xi32>) -> tensor<40x32xi32>
    %706 = tensor.collapse_shape %705 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<40x32xi32> into tensor<1280xi32>
    %707 = tensor.expand_shape %706 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int32"} : tensor<1280xi32> into tensor<2x20x32xi32>
    %708 = tensor.empty() : tensor<2x20x32xbf16>
    %709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707 : tensor<2x20x32xi32>) outs(%708 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "dtype_cast_11", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb86(%710: i32, %711: bf16):
      %712 = arith.sitofp %710 : i32 to bf16
      linalg.yield %712 : bf16
    } -> tensor<2x20x32xbf16>
    %713 = tensor.collapse_shape %523 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<2x20xbf16> into tensor<40xbf16>
    %714 = tensor.expand_shape %713 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 20, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<40xbf16> into tensor<2x20x1xbf16>
    %715 = tensor.empty() : tensor<2x20x32xbf16>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%709, %714 : tensor<2x20x32xbf16>, tensor<2x20x1xbf16>) outs(%715 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb87(%717: bf16, %718: bf16, %719: bf16):
      %720 = arith.mulf %717, %718 : bf16
      linalg.yield %720 : bf16
    } -> tensor<2x20x32xbf16>
    %721 = tensor.collapse_shape %624 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<2x32xbf16> into tensor<64xbf16>
    %722 = tensor.expand_shape %721 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 1, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bfloat16"} : tensor<64xbf16> into tensor<2x1x32xbf16>
    %723 = tensor.empty() : tensor<2x20x32xbf16>
    %724 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716, %722 : tensor<2x20x32xbf16>, tensor<2x1x32xbf16>) outs(%723 : tensor<2x20x32xbf16>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb88(%725: bf16, %726: bf16, %727: bf16):
      %728 = arith.mulf %725, %726 : bf16
      linalg.yield %728 : bf16
    } -> tensor<2x20x32xbf16>
    %729 = tensor.collapse_shape %724 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<2x20x32xbf16> into tensor<1280xbf16>
    %730 = tensor.expand_shape %729 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2, 20, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "bfloat16"} : tensor<1280xbf16> into tensor<1x2x20x32xbf16>
    func.return %730 : tensor<1x2x20x32xbf16>
  }
}
