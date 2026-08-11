builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_dm6zkic2/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>, %1: tensor<16x32xbf16>, %2: tensor<16x32xbf16>) -> tensor<16x32xbf16> {
    %3 = tensor.empty() : tensor<16x32xbf16>
    %4 = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : bf16) outs(%3 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %1 : tensor<16x16xbf16>, tensor<16x32xbf16>) outs(%5 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %7 = tensor.empty() : tensor<16x32xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<16x32xbf16>) outs(%7 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb0(%9: bf16, %10: f32):
      %11 = arith.extf %9 : bf16 to f32
      linalg.yield %11 : f32
    } -> tensor<16x32xf32>
    %12 = tensor.empty() : tensor<16x32xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<16x32xf32>) outs(%12 : tensor<16x32xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb1(%14: f32, %15: f32):
      %16 = arith.constant 1.000000e+00 : f32
      %17 = arith.negf %14 : f32
      %18 = math.exp %17 : f32
      %19 = arith.addf %16, %18 : f32
      %20 = arith.divf %16, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<16x32xf32>
    %21 = tensor.empty() : tensor<16x32xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %13 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%21 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%23: f32, %24: f32, %25: f32):
      %26 = arith.mulf %23, %24 : f32
      linalg.yield %26 : f32
    } -> tensor<16x32xf32>
    %27 = tensor.empty() : tensor<16x32xbf16>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<16x32xf32>) outs(%27 : tensor<16x32xbf16>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%29: f32, %30: bf16):
      %31 = arith.truncf %29 : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<16x32xbf16>
    %32 = tensor.empty() : tensor<16x32xbf16>
    %33 = arith.constant 0.000000e+00 : bf16
    %34 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%33 : bf16) outs(%32 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %35 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %2 : tensor<16x16xbf16>, tensor<16x32xbf16>) outs(%34 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
    %36 = tensor.empty() : tensor<16x32xbf16>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %35 : tensor<16x32xbf16>, tensor<16x32xbf16>) outs(%36 : tensor<16x32xbf16>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb4(%38: bf16, %39: bf16, %40: bf16):
      %41 = arith.mulf %38, %39 : bf16
      linalg.yield %41 : bf16
    } -> tensor<16x32xbf16>
    func.return %37 : tensor<16x32xbf16>
  }
}
