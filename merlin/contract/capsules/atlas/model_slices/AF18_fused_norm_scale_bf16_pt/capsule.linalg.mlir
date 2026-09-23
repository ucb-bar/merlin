builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xbf16>, %1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %2 = tensor.empty() : tensor<32x32xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<32x32xbf16>) outs(%2 : tensor<32x32xbf16>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16):
      %6 = math.rsqrt %4 : bf16
      linalg.yield %6 : bf16
    } -> tensor<32x32xbf16>
    %7 = tensor.empty() : tensor<32x32xbf16>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %3 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%7 : tensor<32x32xbf16>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb1(%9: bf16, %10: bf16, %11: bf16):
      %12 = arith.mulf %9, %10 : bf16
      linalg.yield %12 : bf16
    } -> tensor<32x32xbf16>
    func.return %8 : tensor<32x32xbf16>
  }
}
