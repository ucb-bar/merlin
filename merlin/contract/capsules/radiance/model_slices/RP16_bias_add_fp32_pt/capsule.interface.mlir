builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_t__gguql/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf32>, %1: tensor<16xf32>) -> tensor<16x16xf32> {
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<16x16xf32>, tensor<16xf32>) outs(%2 : tensor<16x16xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32, %6: f32):
      %7 = arith.addf %4, %5 : f32
      linalg.yield %7 : f32
    } -> tensor<16x16xf32>
    func.return %3 : tensor<16x16xf32>
  }
}
