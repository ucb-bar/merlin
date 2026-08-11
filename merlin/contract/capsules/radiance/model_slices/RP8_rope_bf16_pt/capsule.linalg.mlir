builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m__7icctj4/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>) -> tensor<16x16xf32> {
    %1 = tensor.empty() : tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1 : tensor<8xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32"} {
    ^bb0(%3: f32):
      %4 = linalg.index 0 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = arith.sitofp %5 : i64 to f32
      %7 = arith.constant 1.000000e+00 : f32
      %8 = arith.mulf %6, %7 : f32
      %9 = arith.constant 0.000000e+00 : f32
      %10 = arith.addf %9, %8 : f32
      linalg.yield %10 : f32
    } -> tensor<8xf32>
    %11 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 8.000000e+00 : f32
    %12 = tensor.splat %11 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<8xf32>
    %13 = tensor.empty() : tensor<8xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %12 : tensor<8xf32>, tensor<8xf32>) outs(%13 : tensor<8xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%15: f32, %16: f32, %17: f32):
      %18 = arith.divf %15, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<8xf32>
    %19 = tensor.empty() : tensor<8xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%14 : tensor<8xf32>) outs(%19 : tensor<8xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32"} {
    ^bb2(%21: f32, %22: f32):
      %23 = arith.constant 1.000000e+04 : f32
      %24 = math.powf %23, %21 : f32
      linalg.yield %24 : f32
    } -> tensor<8xf32>
    %25 = tensor.empty() : tensor<8xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%20 : tensor<8xf32>) outs(%25 : tensor<8xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb3(%27: f32, %28: f32):
      %29 = arith.constant 1.000000e+00 : f32
      %30 = arith.divf %29, %27 : f32
      linalg.yield %30 : f32
    } -> tensor<8xf32>
    %31 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %32 = tensor.splat %31 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<8xf32>
    %33 = tensor.empty() : tensor<8xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%26, %32 : tensor<8xf32>, tensor<8xf32>) outs(%33 : tensor<8xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%35: f32, %36: f32, %37: f32):
      %38 = arith.mulf %35, %36 : f32
      linalg.yield %38 : f32
    } -> tensor<8xf32>
    %39 = tensor.empty() : tensor<16xf32>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%39 : tensor<16xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32"} {
    ^bb5(%41: f32):
      %42 = linalg.index 0 : index
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.sitofp %43 : i64 to f32
      %45 = arith.constant 1.000000e+00 : f32
      %46 = arith.mulf %44, %45 : f32
      %47 = arith.constant 0.000000e+00 : f32
      %48 = arith.addf %47, %46 : f32
      linalg.yield %48 : f32
    } -> tensor<16xf32>
    %49 = tensor.expand_shape %40 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %50 = tensor.expand_shape %34 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8xf32>
    %51 = tensor.empty() : tensor<16x8xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%49, %50 : tensor<16x1xf32>, tensor<1x8xf32>) outs(%51 : tensor<16x8xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%53: f32, %54: f32, %55: f32):
      %56 = arith.mulf %53, %54 : f32
      linalg.yield %56 : f32
    } -> tensor<16x8xf32>
    %57 = tensor.empty() : tensor<16x8xf32>
    %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<16x8xf32>) outs(%57 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb7(%59: f32, %60: f32):
      %61 = math.cos %59 : f32
      linalg.yield %61 : f32
    } -> tensor<16x8xf32>
    %62 = tensor.empty() : tensor<16x8xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<16x8xf32>) outs(%62 : tensor<16x8xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb8(%64: f32, %65: f32):
      %66 = math.cos %64 : f32
      linalg.yield %66 : f32
    } -> tensor<16x8xf32>
    %67 = tensor.concat dim(1) %58, %63 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %68 = tensor.empty() : tensor<16x8xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<16x8xf32>) outs(%68 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb9(%70: f32, %71: f32):
      %72 = math.sin %70 : f32
      linalg.yield %72 : f32
    } -> tensor<16x8xf32>
    %73 = tensor.empty() : tensor<16x8xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<16x8xf32>) outs(%73 : tensor<16x8xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb10(%75: f32, %76: f32):
      %77 = math.sin %75 : f32
      linalg.yield %77 : f32
    } -> tensor<16x8xf32>
    %78 = tensor.concat dim(1) %69, %74 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x16xf32>
    %79 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 16, 8>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "bfloat16"} : (tensor<16x16xbf16>) -> tensor<16x8xbf16>
    %80 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 8>, static_sizes = array<i64: 16, 8>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "bfloat16"} : (tensor<16x16xbf16>) -> tensor<16x8xbf16>
    %81 = tensor.empty() : tensor<16x16xf32>
    %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %67 : tensor<16x16xbf16>, tensor<16x16xf32>) outs(%81 : tensor<16x16xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb11(%83: bf16, %84: f32, %85: f32):
      %86 = arith.extf %83 : bf16 to f32
      %87 = arith.mulf %86, %84 : f32
      linalg.yield %87 : f32
    } -> tensor<16x16xf32>
    %88 = tensor.empty() : tensor<16x8xbf16>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80 : tensor<16x8xbf16>) outs(%88 : tensor<16x8xbf16>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "bfloat16"} {
    ^bb12(%90: bf16, %91: bf16):
      %92 = arith.negf %90 : bf16
      linalg.yield %92 : bf16
    } -> tensor<16x8xbf16>
    %93 = tensor.concat dim(1) %89, %79 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "bfloat16"} : (tensor<16x8xbf16>, tensor<16x8xbf16>) -> tensor<16x16xbf16>
    %94 = tensor.empty() : tensor<16x16xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%93, %78 : tensor<16x16xbf16>, tensor<16x16xf32>) outs(%94 : tensor<16x16xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb13(%96: bf16, %97: f32, %98: f32):
      %99 = arith.extf %96 : bf16 to f32
      %100 = arith.mulf %99, %97 : f32
      linalg.yield %100 : f32
    } -> tensor<16x16xf32>
    %101 = tensor.empty() : tensor<16x16xf32>
    %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %95 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%101 : tensor<16x16xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb14(%103: f32, %104: f32, %105: f32):
      %106 = arith.addf %103, %104 : f32
      linalg.yield %106 : f32
    } -> tensor<16x16xf32>
    func.return %102 : tensor<16x16xf32>
  }
}
