builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_00widah3/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>) -> tensor<16x16xbf16> {
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xbf16>) outs(%1 : tensor<16x16xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb0(%3: bf16, %4: f32):
      %5 = arith.extf %3 : bf16 to f32
      linalg.yield %5 : f32
    } -> tensor<16x16xf32>
    %6 = tensor.empty() : tensor<16x16xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb1(%8: f32, %9: f32):
      %10 = arith.constant 1.000000e+00 : f32
      %11 = arith.negf %8 : f32
      %12 = math.exp %11 : f32
      %13 = arith.addf %10, %12 : f32
      %14 = arith.divf %10, %13 : f32
      linalg.yield %14 : f32
    } -> tensor<16x16xf32>
    %15 = tensor.empty() : tensor<16x16xf32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %7 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%15 : tensor<16x16xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%17: f32, %18: f32, %19: f32):
      %20 = arith.mulf %17, %18 : f32
      linalg.yield %20 : f32
    } -> tensor<16x16xf32>
    %21 = tensor.empty() : tensor<16x16xbf16>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<16x16xf32>) outs(%21 : tensor<16x16xbf16>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb3(%23: f32, %24: bf16):
      %25 = arith.truncf %23 : f32 to bf16
      linalg.yield %25 : bf16
    } -> tensor<16x16xbf16>
    func.return %22 : tensor<16x16xbf16>
  }
}
