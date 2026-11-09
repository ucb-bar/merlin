builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32x32xf32>, %1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32):
      %6 = arith.constant 5.000000e-01 : f32
      %7 = arith.constant 1.000000e+00 : f32
      %8 = arith.constant 0.707106769 : f32
      %9 = arith.mulf %4, %8 : f32
      %10 = math.erf %9 : f32
      %11 = arith.addf %7, %10 : f32
      %12 = arith.mulf %6, %4 : f32
      %13 = arith.mulf %12, %11 : f32
      linalg.yield %13 : f32
    } -> tensor<32x32xf32>
    %14 = tensor.empty() : tensor<32x32xf32>
    %15 = linalg.transpose ins(%3:tensor<32x32xf32>) outs(%14:tensor<32x32xf32>) permutation = [1, 0]
    %16 = tensor.empty() : tensor<32x32xf32>
    %17 = linalg.transpose ins(%15:tensor<32x32xf32>) outs(%16:tensor<32x32xf32>) permutation = [1, 0]
    %18 = tensor.empty() : tensor<32x32xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb1(%20: f32, %21: f32):
      %22 = arith.constant 2.000000e+00 : f32
      %23 = math.powf %20, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<32x32xf32>
    %24 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %25 = tensor.splat %24 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %26 = linalg.reduce ins(%19:tensor<32x32xf32>) outs(%25:tensor<32xf32>) dimensions = [1]
    (%27: f32, %28: f32) {
      %29 = arith.addf %27, %28 : f32
      linalg.yield %29 : f32
    }
    %30 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %31 = tensor.splat %30 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %32 = tensor.empty() : tensor<32xf32>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%26, %31 : tensor<32xf32>, tensor<32xf32>) outs(%32 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb2(%34: f32, %35: f32, %36: f32):
      %37 = arith.divf %34, %35 : f32
      linalg.yield %37 : f32
    } -> tensor<32xf32>
    %38 = tensor.expand_shape %33 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %39 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %40 = tensor.splat %39 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %41 = tensor.empty() : tensor<32x1xf32>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%38, %40 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%41 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb3(%43: f32, %44: f32, %45: f32):
      %46 = arith.addf %43, %44 : f32
      linalg.yield %46 : f32
    } -> tensor<32x1xf32>
    %47 = tensor.empty() : tensor<32x1xf32>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<32x1xf32>) outs(%47 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb4(%49: f32, %50: f32):
      %51 = math.rsqrt %49 : f32
      linalg.yield %51 : f32
    } -> tensor<32x1xf32>
    %52 = tensor.empty() : tensor<32x32xf32>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %48 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%52 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb5(%54: f32, %55: f32, %56: f32):
      %57 = arith.mulf %54, %55 : f32
      linalg.yield %57 : f32
    } -> tensor<32x32xf32>
    %58 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %59 = tensor.splat %58 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %60 = linalg.reduce ins(%53:tensor<32x32xf32>) outs(%59:tensor<32xf32>) dimensions = [1]
    (%61: f32, %62: f32) {
      %63 = arith.addf %61, %62 : f32
      linalg.yield %63 : f32
    }
    %64 = tensor.expand_shape %60 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %65 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %66 = tensor.splat %65 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %67 = tensor.empty() : tensor<32x1xf32>
    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %66 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%67 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%69: f32, %70: f32, %71: f32):
      %72 = arith.divf %69, %70 : f32
      linalg.yield %72 : f32
    } -> tensor<32x1xf32>
    %73 = tensor.empty() : tensor<32x32xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%53, %68 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%73 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%75: f32, %76: f32, %77: f32):
      %78 = arith.subf %75, %76 : f32
      linalg.yield %78 : f32
    } -> tensor<32x32xf32>
    %79 = tensor.empty() : tensor<32x32xf32>
    %80 = arith.constant 0.000000e+00 : f32
    %81 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%80 : f32) outs(%79 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %82 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%74, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%81 : tensor<32x32xf32>) -> tensor<32x32xf32>
    func.return %82 : tensor<32x32xf32>
  }
}
