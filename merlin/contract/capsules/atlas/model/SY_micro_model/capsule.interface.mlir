builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "float8_weight_only_e4m3"} {
  func.func @forward(%0: tensor<64x64xf32>, %1: tensor<64x64xf32>, %2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %3 = tensor.empty() : tensor<64x64xf32>
    %4 = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : f32) outs(%3 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%2, %0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %7 = tensor.empty() : tensor<64x64xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<64x64xf32>) outs(%7 : tensor<64x64xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb0(%9: f32, %10: f32):
      %11 = arith.constant 2.000000e+00 : f32
      %12 = math.powf %9, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<64x64xf32>
    %13 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %14 = tensor.splat %13 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<64xf32>
    %15 = linalg.reduce ins(%8:tensor<64x64xf32>) outs(%14:tensor<64xf32>) dimensions = [1]
    (%16: f32, %17: f32) {
      %18 = arith.addf %16, %17 : f32
      linalg.yield %18 : f32
    }
    %19 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 6.400000e+01 : f32
    %20 = tensor.splat %19 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<64xf32>
    %21 = tensor.empty() : tensor<64xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%15, %20 : tensor<64xf32>, tensor<64xf32>) outs(%21 : tensor<64xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb1(%23: f32, %24: f32, %25: f32):
      %26 = arith.divf %23, %24 : f32
      linalg.yield %26 : f32
    } -> tensor<64xf32>
    %27 = tensor.expand_shape %22 [[0 : i64, 1 : i64]] output_shape [64, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<64x1xf32>
    %28 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %29 = tensor.splat %28 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<64x1xf32>
    %30 = tensor.empty() : tensor<64x1xf32>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%27, %29 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%30 : tensor<64x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%32: f32, %33: f32, %34: f32):
      %35 = arith.addf %32, %33 : f32
      linalg.yield %35 : f32
    } -> tensor<64x1xf32>
    %36 = tensor.empty() : tensor<64x1xf32>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<64x1xf32>) outs(%36 : tensor<64x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb3(%38: f32, %39: f32):
      %40 = math.rsqrt %38 : f32
      linalg.yield %40 : f32
    } -> tensor<64x1xf32>
    %41 = tensor.empty() : tensor<64x64xf32>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %37 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%41 : tensor<64x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%43: f32, %44: f32, %45: f32):
      %46 = arith.mulf %43, %44 : f32
      linalg.yield %46 : f32
    } -> tensor<64x64xf32>
    %47 = tensor.empty() : tensor<64x64xf32>
    %48 = arith.constant 0.000000e+00 : f32
    %49 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%48 : f32) outs(%47 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %50 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%42, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%49 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %51 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %52 = tensor.splat %51 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<64xf32>
    %53 = linalg.reduce ins(%50:tensor<64x64xf32>) outs(%52:tensor<64xf32>) dimensions = [1]
    (%54: f32, %55: f32) {
      %56 = arith.addf %54, %55 : f32
      linalg.yield %56 : f32
    }
    %57 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [64, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<64x1xf32>
    %58 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 6.400000e+01 : f32
    %59 = tensor.splat %58 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<64x1xf32>
    %60 = tensor.empty() : tensor<64x1xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57, %59 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%60 : tensor<64x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb5(%62: f32, %63: f32, %64: f32):
      %65 = arith.divf %62, %63 : f32
      linalg.yield %65 : f32
    } -> tensor<64x1xf32>
    %66 = tensor.empty() : tensor<64x64xf32>
    %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%50, %61 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%66 : tensor<64x64xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%68: f32, %69: f32, %70: f32):
      %71 = arith.subf %68, %69 : f32
      linalg.yield %71 : f32
    } -> tensor<64x64xf32>
    %72 = tensor.empty() : tensor<64x64xf32>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<64x64xf32>) outs(%72 : tensor<64x64xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb7(%74: f32, %75: f32):
      %76 = arith.constant 5.000000e-01 : f32
      %77 = arith.constant 1.000000e+00 : f32
      %78 = arith.constant 0.707106769 : f32
      %79 = arith.mulf %74, %78 : f32
      %80 = math.erf %79 : f32
      %81 = arith.addf %77, %80 : f32
      %82 = arith.mulf %76, %74 : f32
      %83 = arith.mulf %82, %81 : f32
      linalg.yield %83 : f32
    } -> tensor<64x64xf32>
    %84 = tensor.empty() : tensor<64x64xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73 : tensor<64x64xf32>) outs(%84 : tensor<64x64xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb8(%86: f32, %87: f32):
      %88 = arith.constant 5.000000e-01 : f32
      %89 = arith.constant 1.000000e+00 : f32
      %90 = arith.constant 0.707106769 : f32
      %91 = arith.mulf %86, %90 : f32
      %92 = math.erf %91 : f32
      %93 = arith.addf %89, %92 : f32
      %94 = arith.mulf %88, %86 : f32
      %95 = arith.mulf %94, %93 : f32
      linalg.yield %95 : f32
    } -> tensor<64x64xf32>
    %96 = tensor.empty() : tensor<64x64xf32>
    %97 = linalg.transpose ins(%85:tensor<64x64xf32>) outs(%96:tensor<64x64xf32>) permutation = [1, 0]
    %98 = tensor.empty() : tensor<64x64xf32>
    %99 = linalg.transpose ins(%97:tensor<64x64xf32>) outs(%98:tensor<64x64xf32>) permutation = [1, 0]
    %100 = tensor.empty() : tensor<64x64xf32>
    %101 = linalg.transpose ins(%99:tensor<64x64xf32>) outs(%100:tensor<64x64xf32>) permutation = [1, 0]
    %102 = tensor.empty() : tensor<64x64xf32>
    %103 = linalg.transpose ins(%101:tensor<64x64xf32>) outs(%102:tensor<64x64xf32>) permutation = [1, 0]
    func.return %103 : tensor<64x64xf32>
  }
}
