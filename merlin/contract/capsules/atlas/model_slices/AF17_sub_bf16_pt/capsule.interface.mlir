builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xbf16>, %1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %2 = tensor.empty() : tensor<32x32xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%2 : tensor<32x32xbf16>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16, %6: bf16):
      %7 = arith.subf %4, %5 : bf16
      linalg.yield %7 : bf16
    } -> tensor<32x32xbf16>
    func.return %3 : tensor<32x32xbf16>
  }
}
