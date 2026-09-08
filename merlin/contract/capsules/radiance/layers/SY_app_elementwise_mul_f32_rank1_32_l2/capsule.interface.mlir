builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32xf32>) -> tensor<32xf32> {
    %2 = tensor.empty() : tensor<32xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<32xf32>, tensor<32xf32>) outs(%2 : tensor<32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%4: f32, %5: f32, %6: f32):
      %7 = arith.mulf %4, %5 : f32
      linalg.yield %7 : f32
    } -> tensor<32xf32>
    func.return %3 : tensor<32xf32>
  }
}
