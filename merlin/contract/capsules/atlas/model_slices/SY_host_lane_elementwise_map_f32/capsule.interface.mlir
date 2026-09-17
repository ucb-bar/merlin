builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %1 = tensor.empty() : tensor<32x64xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<32x64xf32>) outs(%1 : tensor<32x64xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb0(%3: f32, %4: f32):
      %5 = arith.constant 5.000000e-01 : f32
      %6 = arith.constant 1.000000e+00 : f32
      %7 = arith.constant 0.707106769 : f32
      %8 = arith.mulf %3, %7 : f32
      %9 = math.erf %8 : f32
      %10 = arith.addf %6, %9 : f32
      %11 = arith.mulf %5, %3 : f32
      %12 = arith.mulf %11, %10 : f32
      linalg.yield %12 : f32
    } -> tensor<32x64xf32>
    func.return %2 : tensor<32x64xf32>
  }
}
