builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32xf32>, %2: tensor<32x32xi8>, %3: tensor<32x32xi8>, %4: tensor<16x32xi8>) -> tensor<16x32xi8> {
    %5 = tensor.empty() : tensor<16x32xi8>
    %6 = arith.constant 0 : i8
    %7 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%6 : i8) outs(%5 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %8 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%4, %2 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%7 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %9 = tensor.empty() : tensor<16x32xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<16x32xi8>) outs(%9 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb0(%11: i8, %12: f32):
      %13 = arith.sitofp %11 : i8 to f32
      linalg.yield %13 : f32
    } -> tensor<16x32xf32>
    %14 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} 0.000000e+00 : f32
    %15 = tensor.splat %14 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32>
    %16 = linalg.reduce ins(%10:tensor<16x32xf32>) outs(%15:tensor<16xf32>) dimensions = [1]
    (%17: f32, %18: f32) {
      %19 = arith.addf %17, %18 : f32
      linalg.yield %19 : f32
    }
    %20 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} 3.200000e+01 : f32
    %21 = tensor.splat %20 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32>
    %22 = tensor.empty() : tensor<16xf32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%16, %21 : tensor<16xf32>, tensor<16xf32>) outs(%22 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb1(%24: f32, %25: f32, %26: f32):
      %27 = arith.divf %24, %25 : f32
      linalg.yield %27 : f32
    } -> tensor<16xf32>
    %28 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32> into tensor<16x1xf32>
    %29 = tensor.empty() : tensor<16x32xf32>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %28 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%29 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb2(%31: f32, %32: f32, %33: f32):
      %34 = arith.subf %31, %32 : f32
      linalg.yield %34 : f32
    } -> tensor<16x32xf32>
    %35 = tensor.empty() : tensor<16x32xf32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30, %30 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%35 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb3(%37: f32, %38: f32, %39: f32):
      %40 = arith.mulf %37, %38 : f32
      linalg.yield %40 : f32
    } -> tensor<16x32xf32>
    %41 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} 0.000000e+00 : f32
    %42 = tensor.splat %41 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32>
    %43 = linalg.reduce ins(%36:tensor<16x32xf32>) outs(%42:tensor<16xf32>) dimensions = [1]
    (%44: f32, %45: f32) {
      %46 = arith.addf %44, %45 : f32
      linalg.yield %46 : f32
    }
    %47 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} 3.200000e+01 : f32
    %48 = tensor.splat %47 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32>
    %49 = tensor.empty() : tensor<16xf32>
    %50 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43, %48 : tensor<16xf32>, tensor<16xf32>) outs(%49 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb4(%51: f32, %52: f32, %53: f32):
      %54 = arith.divf %51, %52 : f32
      linalg.yield %54 : f32
    } -> tensor<16xf32>
    %55 = tensor.expand_shape %50 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16xf32> into tensor<16x1xf32>
    %56 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} 1.000000e-05 : f32
    %57 = tensor.splat %56 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} : tensor<16x1xf32>
    %58 = tensor.empty() : tensor<16x1xf32>
    %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%55, %57 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%58 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb5(%60: f32, %61: f32, %62: f32):
      %63 = arith.addf %60, %61 : f32
      linalg.yield %63 : f32
    } -> tensor<16x1xf32>
    %64 = tensor.empty() : tensor<16x1xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<16x1xf32>) outs(%64 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb6(%66: f32, %67: f32):
      %68 = math.rsqrt %66 : f32
      linalg.yield %68 : f32
    } -> tensor<16x1xf32>
    %69 = tensor.empty() : tensor<16x32xf32>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30, %65 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%69 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb7(%71: f32, %72: f32, %73: f32):
      %74 = arith.mulf %71, %72 : f32
      linalg.yield %74 : f32
    } -> tensor<16x32xf32>
    %75 = tensor.empty() : tensor<16x32xf32>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %0 : tensor<16x32xf32>, tensor<32xf32>) outs(%75 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb8(%77: f32, %78: f32, %79: f32):
      %80 = arith.mulf %77, %78 : f32
      linalg.yield %80 : f32
    } -> tensor<16x32xf32>
    %81 = tensor.empty() : tensor<16x32xf32>
    %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %1 : tensor<16x32xf32>, tensor<32xf32>) outs(%81 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln", prov.fqn = "ln"} {
    ^bb9(%83: f32, %84: f32, %85: f32):
      %86 = arith.addf %83, %84 : f32
      linalg.yield %86 : f32
    } -> tensor<16x32xf32>
    %87 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.100000e+00 : f32
    %88 = tensor.splat %87 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %89 = tensor.empty() : tensor<16x32xf32>
    %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %88 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%89 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%91: f32, %92: f32, %93: f32):
      %94 = arith.mulf %91, %92 : f32
      linalg.yield %94 : f32
    } -> tensor<16x32xf32>
    %95 = tensor.empty() : tensor<16x32xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%90 : tensor<16x32xf32>) outs(%95 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb11(%97: f32, %98: f32):
      %99 = math.tanh %97 : f32
      linalg.yield %99 : f32
    } -> tensor<16x32xf32>
    %100 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %101 = tensor.splat %100 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %102 = tensor.empty() : tensor<16x32xf32>
    %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%96, %101 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%102 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb12(%104: f32, %105: f32, %106: f32):
      %107 = arith.mulf %104, %105 : f32
      linalg.yield %107 : f32
    } -> tensor<16x32xf32>
    %108 = tensor.empty() : tensor<16x32xi8>
    %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%103 : tensor<16x32xf32>) outs(%108 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb13(%110: f32, %111: i8):
      %112 = arith.fptosi %110 : f32 to i8
      linalg.yield %112 : i8
    } -> tensor<16x32xi8>
    %113 = tensor.empty() : tensor<16x32xi8>
    %114 = arith.constant 0 : i8
    %115 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%114 : i8) outs(%113 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %116 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%109, %3 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%115 : tensor<16x32xi8>) -> tensor<16x32xi8>
    func.return %116 : tensor<16x32xi8>
  }
}
