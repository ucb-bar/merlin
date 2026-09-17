builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xbf16>, %1: tensor<16x16xbf16>, %2: tensor<16xbf16>) -> tensor<16x16xbf16> {
    %3 = tensor.empty() : tensor<16x16xbf16>
    %4 = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : bf16) outs(%3 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %1 : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%5 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %7 = tensor.empty() : tensor<16x16xbf16>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %2 : tensor<16x16xbf16>, tensor<16xbf16>) outs(%7 : tensor<16x16xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%9: bf16, %10: bf16, %11: bf16):
      %12 = arith.addf %9, %10 : bf16
      linalg.yield %12 : bf16
    } -> tensor<16x16xbf16>
    func.return %8 : tensor<16x16xbf16>
  }
}
