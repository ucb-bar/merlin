builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xf16>) -> tensor<16x32xf16> {
    %1 = tensor.empty() : tensor<16x32xf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x32xf16>) outs(%1 : tensor<16x32xf16>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float16"} {
    ^bb0(%3: f16, %4: f16):
      %5 = arith.constant 5.000000e-01 : f16
      %6 = arith.constant 1.000000e+00 : f16
      %7 = arith.constant 7.070310e-01 : f16
      %8 = arith.mulf %3, %7 : f16
      %9 = math.erf %8 : f16
      %10 = arith.addf %6, %9 : f16
      %11 = arith.mulf %5, %3 : f16
      %12 = arith.mulf %11, %10 : f16
      linalg.yield %12 : f16
    } -> tensor<16x32xf16>
    func.return %2 : tensor<16x32xf16>
  }
}
