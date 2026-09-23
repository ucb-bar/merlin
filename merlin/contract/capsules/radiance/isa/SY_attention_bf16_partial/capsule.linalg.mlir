builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x31xbf16>, %1: tensor<15x31xbf16>, %2: tensor<15x31xbf16>) -> tensor<16x31xbf16> {
    %3 = tensor.empty() : tensor<31x15xbf16>
    %4 = linalg.transpose ins(%1:tensor<15x31xbf16>) outs(%3:tensor<31x15xbf16>) permutation = [1, 0]
    %5 = tensor.empty() : tensor<16x15xbf16>
    %6 = arith.constant 0.000000e+00 : bf16
    %7 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%6 : bf16) outs(%5 : tensor<16x15xbf16>) -> tensor<16x15xbf16>
    %8 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16", prov.transposed_b = "true"} ins(%0, %4 : tensor<16x31xbf16>, tensor<31x15xbf16>) outs(%7 : tensor<16x15xbf16>) -> tensor<16x15xbf16>
    %9 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 5.562500e+00 : bf16
    %10 = tensor.splat %9 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<16x15xbf16>
    %11 = tensor.empty() : tensor<16x15xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %10 : tensor<16x15xbf16>, tensor<16x15xbf16>) outs(%11 : tensor<16x15xbf16>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%13: bf16, %14: bf16, %15: bf16):
      %16 = arith.divf %13, %14 : bf16
      linalg.yield %16 : bf16
    } -> tensor<16x15xbf16>
    %17 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %18 = tensor.splat %17 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %19 = linalg.reduce ins(%12:tensor<16x15xbf16>) outs(%18:tensor<16xbf16>) dimensions = [1]
    (%20: bf16, %21: bf16) {
      %22 = arith.maximumf %20, %21 : bf16
      linalg.yield %22 : bf16
    }
    %23 = tensor.expand_shape %19 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %24 = tensor.empty() : tensor<16x15xbf16>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %23 : tensor<16x15xbf16>, tensor<16x1xbf16>) outs(%24 : tensor<16x15xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%26: bf16, %27: bf16, %28: bf16):
      %29 = arith.subf %26, %27 : bf16
      linalg.yield %29 : bf16
    } -> tensor<16x15xbf16>
    %30 = tensor.empty() : tensor<16x15xbf16>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<16x15xbf16>) outs(%30 : tensor<16x15xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%32: bf16, %33: bf16):
      %34 = math.exp %32 : bf16
      linalg.yield %34 : bf16
    } -> tensor<16x15xbf16>
    %35 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %36 = tensor.splat %35 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %37 = linalg.reduce ins(%31:tensor<16x15xbf16>) outs(%36:tensor<16xbf16>) dimensions = [1]
    (%38: bf16, %39: bf16) {
      %40 = arith.addf %38, %39 : bf16
      linalg.yield %40 : bf16
    }
    %41 = tensor.expand_shape %37 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %42 = tensor.empty() : tensor<16x15xbf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41 : tensor<16x15xbf16>, tensor<16x1xbf16>) outs(%42 : tensor<16x15xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%44: bf16, %45: bf16, %46: bf16):
      %47 = arith.divf %44, %45 : bf16
      linalg.yield %47 : bf16
    } -> tensor<16x15xbf16>
    %48 = tensor.empty() : tensor<16x31xbf16>
    %49 = arith.constant 0.000000e+00 : bf16
    %50 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%49 : bf16) outs(%48 : tensor<16x31xbf16>) -> tensor<16x31xbf16>
    %51 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%43, %2 : tensor<16x15xbf16>, tensor<15x31xbf16>) outs(%50 : tensor<16x31xbf16>) -> tensor<16x31xbf16>
    func.return %51 : tensor<16x31xbf16>
  }
}
