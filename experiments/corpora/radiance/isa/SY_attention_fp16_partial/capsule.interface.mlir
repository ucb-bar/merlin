builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x31xf16>, %1: tensor<15x31xf16>, %2: tensor<15x31xf16>) -> tensor<16x31xf16> {
    %3 = tensor.empty() : tensor<31x15xf16>
    %4 = linalg.transpose ins(%1:tensor<15x31xf16>) outs(%3:tensor<31x15xf16>) permutation = [1, 0]
    %5 = tensor.empty() : tensor<16x15xf16>
    %6 = arith.constant 0.000000e+00 : f16
    %7 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%6 : f16) outs(%5 : tensor<16x15xf16>) -> tensor<16x15xf16>
    %8 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float16", prov.transposed_b = "true"} ins(%0, %4 : tensor<16x31xf16>, tensor<31x15xf16>) outs(%7 : tensor<16x15xf16>) -> tensor<16x15xf16>
    %9 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float16"} 5.566410e+00 : f16
    %10 = tensor.splat %9 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float16"} : tensor<16x15xf16>
    %11 = tensor.empty() : tensor<16x15xf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %10 : tensor<16x15xf16>, tensor<16x15xf16>) outs(%11 : tensor<16x15xf16>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float16"} {
    ^bb0(%13: f16, %14: f16, %15: f16):
      %16 = arith.divf %13, %14 : f16
      linalg.yield %16 : f16
    } -> tensor<16x15xf16>
    %17 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} 0xfc00 : f16
    %18 = tensor.splat %17 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<16xf16>
    %19 = linalg.reduce ins(%12:tensor<16x15xf16>) outs(%18:tensor<16xf16>) dimensions = [1]
    (%20: f16, %21: f16) {
      %22 = arith.maximumf %20, %21 : f16
      linalg.yield %22 : f16
    }
    %23 = tensor.expand_shape %19 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<16xf16> into tensor<16x1xf16>
    %24 = tensor.empty() : tensor<16x15xf16>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %23 : tensor<16x15xf16>, tensor<16x1xf16>) outs(%24 : tensor<16x15xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb1(%26: f16, %27: f16, %28: f16):
      %29 = arith.subf %26, %27 : f16
      linalg.yield %29 : f16
    } -> tensor<16x15xf16>
    %30 = tensor.empty() : tensor<16x15xf16>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<16x15xf16>) outs(%30 : tensor<16x15xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb2(%32: f16, %33: f16):
      %34 = math.exp %32 : f16
      linalg.yield %34 : f16
    } -> tensor<16x15xf16>
    %35 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} 0.000000e+00 : f16
    %36 = tensor.splat %35 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<16xf16>
    %37 = linalg.reduce ins(%31:tensor<16x15xf16>) outs(%36:tensor<16xf16>) dimensions = [1]
    (%38: f16, %39: f16) {
      %40 = arith.addf %38, %39 : f16
      linalg.yield %40 : f16
    }
    %41 = tensor.expand_shape %37 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} : tensor<16xf16> into tensor<16x1xf16>
    %42 = tensor.empty() : tensor<16x15xf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41 : tensor<16x15xf16>, tensor<16x1xf16>) outs(%42 : tensor<16x15xf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float16"} {
    ^bb3(%44: f16, %45: f16, %46: f16):
      %47 = arith.divf %44, %45 : f16
      linalg.yield %47 : f16
    } -> tensor<16x15xf16>
    %48 = tensor.empty() : tensor<16x31xf16>
    %49 = arith.constant 0.000000e+00 : f16
    %50 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%49 : f16) outs(%48 : tensor<16x31xf16>) -> tensor<16x31xf16>
    %51 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float16"} ins(%43, %2 : tensor<16x15xf16>, tensor<15x31xf16>) outs(%50 : tensor<16x31xf16>) -> tensor<16x31xf16>
    func.return %51 : tensor<16x31xf16>
  }
}
