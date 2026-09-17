builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>, %1: tensor<16xbf16>) -> tensor<16x16xbf16> {
    %2 = tensor.empty() : tensor<16x16xbf16>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<16x16xbf16>, tensor<16xbf16>) outs(%2 : tensor<16x16xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%4: bf16, %5: bf16, %6: bf16):
      %7 = arith.addf %4, %5 : bf16
      linalg.yield %7 : bf16
    } -> tensor<16x16xbf16>
    func.return %3 : tensor<16x16xbf16>
  }
}
