builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %1 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.000000e+01 : f32
    %2 = tensor.splat %1 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<16x16xf32>
    %3 = tensor.empty() : tensor<16x16xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%5: f32, %6: f32, %7: f32):
      %8 = arith.divf %5, %6 : f32
      linalg.yield %8 : f32
    } -> tensor<16x16xf32>
    %9 = tensor.empty() : tensor<16x16xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<16x16xf32>) outs(%9 : tensor<16x16xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb1(%11: f32, %12: f32):
      %13 = math.tanh %11 : f32
      linalg.yield %13 : f32
    } -> tensor<16x16xf32>
    %14 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 5.000000e+01 : f32
    %15 = tensor.splat %14 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x16xf32>
    %16 = tensor.empty() : tensor<16x16xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %15 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%16 : tensor<16x16xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%18: f32, %19: f32, %20: f32):
      %21 = arith.mulf %18, %19 : f32
      linalg.yield %21 : f32
    } -> tensor<16x16xf32>
    func.return %17 : tensor<16x16xf32>
  }
}
