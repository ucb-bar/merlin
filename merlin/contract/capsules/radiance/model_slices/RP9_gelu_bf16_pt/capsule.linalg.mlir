builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>) -> tensor<16x16xbf16> {
    %1 = tensor.empty() : tensor<16x16xbf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xbf16>) outs(%1 : tensor<16x16xbf16>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%3: bf16, %4: bf16):
      %5 = arith.constant 5.000000e-01 : bf16
      %6 = arith.constant 1.000000e+00 : bf16
      %7 = arith.constant 7.070310e-01 : bf16
      %8 = arith.mulf %3, %7 : bf16
      %9 = math.erf %8 : bf16
      %10 = arith.addf %6, %9 : bf16
      %11 = arith.mulf %5, %3 : bf16
      %12 = arith.mulf %11, %10 : bf16
      linalg.yield %12 : bf16
    } -> tensor<16x16xbf16>
    func.return %2 : tensor<16x16xbf16>
  }
}
