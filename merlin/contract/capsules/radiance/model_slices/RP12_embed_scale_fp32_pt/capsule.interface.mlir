builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_4avw_g5n/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %1 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 4.000000e+00 : f32
    %2 = tensor.splat %1 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x16xf32>
    %3 = tensor.empty() : tensor<16x16xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %2 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%3 : tensor<16x16xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%5: f32, %6: f32, %7: f32):
      %8 = arith.mulf %5, %6 : f32
      linalg.yield %8 : f32
    } -> tensor<16x16xf32>
    func.return %4 : tensor<16x16xf32>
  }
}
