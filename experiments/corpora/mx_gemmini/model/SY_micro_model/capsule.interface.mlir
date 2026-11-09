builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32x32xf32>, %2: tensor<32x32xf32>, %3: tensor<32x32xf32>, %4: tensor<32x32xf32>, %5: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = arith.constant 0.000000e+00 : f32
    %8 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%7 : f32) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%5, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %10 = tensor.empty() : tensor<32x32xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<32x32xf32>) outs(%10 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb0(%12: f32, %13: f32):
      %14 = arith.constant 5.000000e-01 : f32
      %15 = arith.constant 1.000000e+00 : f32
      %16 = arith.constant 0.707106769 : f32
      %17 = arith.mulf %12, %16 : f32
      %18 = math.erf %17 : f32
      %19 = arith.addf %15, %18 : f32
      %20 = arith.mulf %14, %12 : f32
      %21 = arith.mulf %20, %19 : f32
      linalg.yield %21 : f32
    } -> tensor<32x32xf32>
    %22 = tensor.empty() : tensor<32x32xf32>
    %23 = arith.constant 0.000000e+00 : f32
    %24 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%23 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %25 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%11, %1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %26 = tensor.empty() : tensor<32x32xf32>
    %27 = linalg.transpose ins(%25:tensor<32x32xf32>) outs(%26:tensor<32x32xf32>) permutation = [1, 0]
    %28 = tensor.empty() : tensor<32x32xf32>
    %29 = linalg.transpose ins(%27:tensor<32x32xf32>) outs(%28:tensor<32x32xf32>) permutation = [1, 0]
    %30 = tensor.empty() : tensor<32x32xf32>
    %31 = arith.constant 0.000000e+00 : f32
    %32 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%31 : f32) outs(%30 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %33 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%29, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%32 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %34 = tensor.empty() : tensor<32x32xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<32x32xf32>) outs(%34 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb1(%36: f32, %37: f32):
      %38 = arith.constant 2.000000e+00 : f32
      %39 = math.powf %36, %38 : f32
      linalg.yield %39 : f32
    } -> tensor<32x32xf32>
    %40 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %41 = tensor.splat %40 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %42 = linalg.reduce ins(%35:tensor<32x32xf32>) outs(%41:tensor<32xf32>) dimensions = [1]
    (%43: f32, %44: f32) {
      %45 = arith.addf %43, %44 : f32
      linalg.yield %45 : f32
    }
    %46 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %47 = tensor.splat %46 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %48 = tensor.empty() : tensor<32xf32>
    %49 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%42, %47 : tensor<32xf32>, tensor<32xf32>) outs(%48 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb2(%50: f32, %51: f32, %52: f32):
      %53 = arith.divf %50, %51 : f32
      linalg.yield %53 : f32
    } -> tensor<32xf32>
    %54 = tensor.expand_shape %49 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %55 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %56 = tensor.splat %55 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %57 = tensor.empty() : tensor<32x1xf32>
    %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%54, %56 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%57 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb3(%59: f32, %60: f32, %61: f32):
      %62 = arith.addf %59, %60 : f32
      linalg.yield %62 : f32
    } -> tensor<32x1xf32>
    %63 = tensor.empty() : tensor<32x1xf32>
    %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58 : tensor<32x1xf32>) outs(%63 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb4(%65: f32, %66: f32):
      %67 = math.rsqrt %65 : f32
      linalg.yield %67 : f32
    } -> tensor<32x1xf32>
    %68 = tensor.empty() : tensor<32x32xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %64 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%68 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb5(%70: f32, %71: f32, %72: f32):
      %73 = arith.mulf %70, %71 : f32
      linalg.yield %73 : f32
    } -> tensor<32x32xf32>
    %74 = tensor.empty() : tensor<32x32xf32>
    %75 = arith.constant 0.000000e+00 : f32
    %76 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%75 : f32) outs(%74 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %77 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%69, %3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%76 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %78 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %79 = tensor.splat %78 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %80 = linalg.reduce ins(%77:tensor<32x32xf32>) outs(%79:tensor<32xf32>) dimensions = [1]
    (%81: f32, %82: f32) {
      %83 = arith.addf %81, %82 : f32
      linalg.yield %83 : f32
    }
    %84 = tensor.expand_shape %80 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %85 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %86 = tensor.splat %85 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %87 = tensor.empty() : tensor<32x1xf32>
    %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%84, %86 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%87 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%89: f32, %90: f32, %91: f32):
      %92 = arith.divf %89, %90 : f32
      linalg.yield %92 : f32
    } -> tensor<32x1xf32>
    %93 = tensor.empty() : tensor<32x32xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%77, %88 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%93 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%95: f32, %96: f32, %97: f32):
      %98 = arith.subf %95, %96 : f32
      linalg.yield %98 : f32
    } -> tensor<32x32xf32>
    %99 = tensor.empty() : tensor<32x32xf32>
    %100 = arith.constant 0.000000e+00 : f32
    %101 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%100 : f32) outs(%99 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %102 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%94, %4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%101 : tensor<32x32xf32>) -> tensor<32x32xf32>
    func.return %102 : tensor<32x32xf32>
  }
}
