builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xf32>, %1: tensor<16x32xf32>, %2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %3 = tensor.empty() : tensor<32x16xf32>
    %4 = linalg.transpose ins(%1:tensor<16x32xf32>) outs(%3:tensor<32x16xf32>) permutation = [1, 0]
    %5 = tensor.empty() : tensor<16x16xf32>
    %6 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%6 : f32) outs(%5 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %8 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%0, %4 : tensor<16x32xf32>, tensor<32x16xf32>) outs(%7 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %9 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
    %10 = tensor.splat %9 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<16x16xf32>
    %11 = tensor.empty() : tensor<16x16xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %10 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%11 : tensor<16x16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%13: f32, %14: f32, %15: f32):
      %16 = arith.divf %13, %14 : f32
      linalg.yield %16 : f32
    } -> tensor<16x16xf32>
    %17 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %18 = tensor.splat %17 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %19 = linalg.reduce ins(%12:tensor<16x16xf32>) outs(%18:tensor<16xf32>) dimensions = [1]
    (%20: f32, %21: f32) {
      %22 = arith.maximumf %20, %21 : f32
      linalg.yield %22 : f32
    }
    %23 = tensor.expand_shape %19 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %24 = tensor.empty() : tensor<16x16xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %23 : tensor<16x16xf32>, tensor<16x1xf32>) outs(%24 : tensor<16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb1(%26: f32, %27: f32, %28: f32):
      %29 = arith.subf %26, %27 : f32
      linalg.yield %29 : f32
    } -> tensor<16x16xf32>
    %30 = tensor.empty() : tensor<16x16xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<16x16xf32>) outs(%30 : tensor<16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%32: f32, %33: f32):
      %34 = math.exp %32 : f32
      linalg.yield %34 : f32
    } -> tensor<16x16xf32>
    %35 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %36 = tensor.splat %35 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %37 = linalg.reduce ins(%31:tensor<16x16xf32>) outs(%36:tensor<16xf32>) dimensions = [1]
    (%38: f32, %39: f32) {
      %40 = arith.addf %38, %39 : f32
      linalg.yield %40 : f32
    }
    %41 = tensor.expand_shape %37 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %42 = tensor.empty() : tensor<16x16xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41 : tensor<16x16xf32>, tensor<16x1xf32>) outs(%42 : tensor<16x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%44: f32, %45: f32, %46: f32):
      %47 = arith.divf %44, %45 : f32
      linalg.yield %47 : f32
    } -> tensor<16x16xf32>
    %48 = tensor.empty() : tensor<16x32xf32>
    %49 = arith.constant 0.000000e+00 : f32
    %50 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%49 : f32) outs(%48 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %51 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%43, %2 : tensor<16x16xf32>, tensor<16x32xf32>) outs(%50 : tensor<16x32xf32>) -> tensor<16x32xf32>
    func.return %51 : tensor<16x32xf32>
  }
}
