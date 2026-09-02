builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x64xf32>, %1: tensor<1x64xf32>) -> tensor<32x64xf32> {
    %2 = tensor.empty() : tensor<32x64xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<32x64xf32>) outs(%2 : tensor<32x64xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32):
      %6 = arith.constant 2.000000e+00 : f32
      %7 = math.powf %4, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<32x64xf32>
    %8 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %9 = tensor.splat %8 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %10 = linalg.reduce ins(%3:tensor<32x64xf32>) outs(%9:tensor<32xf32>) dimensions = [1]
    (%11: f32, %12: f32) {
      %13 = arith.addf %11, %12 : f32
      linalg.yield %13 : f32
    }
    %14 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 6.400000e+01 : f32
    %15 = tensor.splat %14 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %16 = tensor.empty() : tensor<32xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %15 : tensor<32xf32>, tensor<32xf32>) outs(%16 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb1(%18: f32, %19: f32, %20: f32):
      %21 = arith.divf %18, %19 : f32
      linalg.yield %21 : f32
    } -> tensor<32xf32>
    %22 = tensor.expand_shape %17 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %23 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.52587891e-05 : f32
    %24 = tensor.splat %23 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %25 = tensor.empty() : tensor<32x1xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %24 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%25 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%27: f32, %28: f32, %29: f32):
      %30 = arith.addf %27, %28 : f32
      linalg.yield %30 : f32
    } -> tensor<32x1xf32>
    %31 = tensor.empty() : tensor<32x1xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<32x1xf32>) outs(%31 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb3(%33: f32, %34: f32):
      %35 = math.rsqrt %33 : f32
      linalg.yield %35 : f32
    } -> tensor<32x1xf32>
    %36 = tensor.empty() : tensor<32x64xf32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %32 : tensor<32x64xf32>, tensor<32x1xf32>) outs(%36 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%38: f32, %39: f32, %40: f32):
      %41 = arith.mulf %38, %39 : f32
      linalg.yield %41 : f32
    } -> tensor<32x64xf32>
    %42 = tensor.empty() : tensor<32x64xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %1 : tensor<32x64xf32>, tensor<1x64xf32>) outs(%42 : tensor<32x64xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb5(%44: f32, %45: f32, %46: f32):
      %47 = arith.mulf %44, %45 : f32
      linalg.yield %47 : f32
    } -> tensor<32x64xf32>
    func.return %43 : tensor<32x64xf32>
  }
}
