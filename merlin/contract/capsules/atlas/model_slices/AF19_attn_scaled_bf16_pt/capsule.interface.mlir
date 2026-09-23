builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xbf16>, %1: tensor<32x32xbf16>, %2: tensor<32x32xbf16>, %3: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %4 = tensor.empty() : tensor<32x32xbf16>
    %5 = arith.constant 0.000000e+00 : bf16
    %6 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%5 : bf16) outs(%4 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%6 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = tensor.empty() : tensor<32x32xbf16>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %3 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%8 : tensor<32x32xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%10: bf16, %11: bf16, %12: bf16):
      %13 = arith.mulf %10, %11 : bf16
      linalg.yield %13 : bf16
    } -> tensor<32x32xbf16>
    %14 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %15 = tensor.splat %14 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<32xbf16>
    %16 = linalg.reduce ins(%9:tensor<32x32xbf16>) outs(%15:tensor<32xbf16>) dimensions = [1]
    (%17: bf16, %18: bf16) {
      %19 = arith.maximumf %17, %18 : bf16
      linalg.yield %19 : bf16
    }
    %20 = tensor.expand_shape %16 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<32xbf16> into tensor<32x1xbf16>
    %21 = tensor.empty() : tensor<32x32xbf16>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %20 : tensor<32x32xbf16>, tensor<32x1xbf16>) outs(%21 : tensor<32x32xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb1(%23: bf16, %24: bf16, %25: bf16):
      %26 = arith.subf %23, %24 : bf16
      linalg.yield %26 : bf16
    } -> tensor<32x32xbf16>
    %27 = tensor.empty() : tensor<32x32xbf16>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<32x32xbf16>) outs(%27 : tensor<32x32xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%29: bf16, %30: bf16):
      %31 = math.exp %29 : bf16
      linalg.yield %31 : bf16
    } -> tensor<32x32xbf16>
    %32 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %33 = tensor.splat %32 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<32xbf16>
    %34 = linalg.reduce ins(%28:tensor<32x32xbf16>) outs(%33:tensor<32xbf16>) dimensions = [1]
    (%35: bf16, %36: bf16) {
      %37 = arith.addf %35, %36 : bf16
      linalg.yield %37 : bf16
    }
    %38 = tensor.expand_shape %34 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<32xbf16> into tensor<32x1xbf16>
    %39 = tensor.empty() : tensor<32x32xbf16>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %38 : tensor<32x32xbf16>, tensor<32x1xbf16>) outs(%39 : tensor<32x32xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%41: bf16, %42: bf16, %43: bf16):
      %44 = arith.divf %41, %42 : bf16
      linalg.yield %44 : bf16
    } -> tensor<32x32xbf16>
    %45 = tensor.empty() : tensor<32x32xbf16>
    %46 = arith.constant 0.000000e+00 : bf16
    %47 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%46 : bf16) outs(%45 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %48 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%40, %2 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%47 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    func.return %48 : tensor<32x32xbf16>
  }
}
