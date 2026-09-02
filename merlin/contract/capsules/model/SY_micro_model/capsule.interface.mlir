builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%1, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb0(%8: f32, %9: f32):
      %10 = arith.constant 2.000000e+00 : f32
      %11 = math.powf %8, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<32x32xf32>
    %12 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %13 = tensor.splat %12 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %14 = linalg.reduce ins(%7:tensor<32x32xf32>) outs(%13:tensor<32xf32>) dimensions = [1]
    (%15: f32, %16: f32) {
      %17 = arith.addf %15, %16 : f32
      linalg.yield %17 : f32
    }
    %18 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %19 = tensor.splat %18 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %20 = tensor.empty() : tensor<32xf32>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%14, %19 : tensor<32xf32>, tensor<32xf32>) outs(%20 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb1(%22: f32, %23: f32, %24: f32):
      %25 = arith.divf %22, %23 : f32
      linalg.yield %25 : f32
    } -> tensor<32xf32>
    %26 = tensor.expand_shape %21 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %27 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %28 = tensor.splat %27 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %29 = tensor.empty() : tensor<32x1xf32>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26, %28 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%29 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%31: f32, %32: f32, %33: f32):
      %34 = arith.addf %31, %32 : f32
      linalg.yield %34 : f32
    } -> tensor<32x1xf32>
    %35 = tensor.empty() : tensor<32x1xf32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30 : tensor<32x1xf32>) outs(%35 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb3(%37: f32, %38: f32):
      %39 = math.rsqrt %37 : f32
      linalg.yield %39 : f32
    } -> tensor<32x1xf32>
    %40 = tensor.empty() : tensor<32x32xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %36 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%40 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%42: f32, %43: f32, %44: f32):
      %45 = arith.mulf %42, %43 : f32
      linalg.yield %45 : f32
    } -> tensor<32x32xf32>
    %46 = tensor.empty() : tensor<32x32xf32>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%41 : tensor<32x32xf32>) outs(%46 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb5(%48: f32, %49: f32):
      %50 = arith.constant 5.000000e-01 : f32
      %51 = arith.constant 1.000000e+00 : f32
      %52 = arith.constant 0.707106769 : f32
      %53 = arith.mulf %48, %52 : f32
      %54 = math.erf %53 : f32
      %55 = arith.addf %51, %54 : f32
      %56 = arith.mulf %50, %48 : f32
      %57 = arith.mulf %56, %55 : f32
      linalg.yield %57 : f32
    } -> tensor<32x32xf32>
    %58 = tensor.empty() : tensor<32x32xf32>
    %59 = linalg.transpose ins(%47:tensor<32x32xf32>) outs(%58:tensor<32x32xf32>) permutation = [1, 0]
    %60 = tensor.empty() : tensor<32x32xf32>
    %61 = linalg.transpose ins(%59:tensor<32x32xf32>) outs(%60:tensor<32x32xf32>) permutation = [1, 0]
    %62 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %63 = tensor.splat %62 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %64 = linalg.reduce ins(%61:tensor<32x32xf32>) outs(%63:tensor<32xf32>) dimensions = [1]
    (%65: f32, %66: f32) {
      %67 = arith.addf %65, %66 : f32
      linalg.yield %67 : f32
    }
    %68 = tensor.expand_shape %64 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %69 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %70 = tensor.splat %69 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %71 = tensor.empty() : tensor<32x1xf32>
    %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %70 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%71 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%73: f32, %74: f32, %75: f32):
      %76 = arith.divf %73, %74 : f32
      linalg.yield %76 : f32
    } -> tensor<32x1xf32>
    %77 = tensor.empty() : tensor<32x32xf32>
    %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %72 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%77 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%79: f32, %80: f32, %81: f32):
      %82 = arith.subf %79, %80 : f32
      linalg.yield %82 : f32
    } -> tensor<32x32xf32>
    func.return %78 : tensor<32x32xf32>
  }
}
