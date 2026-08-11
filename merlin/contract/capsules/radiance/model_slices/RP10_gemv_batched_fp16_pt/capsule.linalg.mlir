builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_j6avpgi8/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x16x16xf16>, %1: tensor<2x16x1xf16>) -> tensor<2x16x1xf16> {
    %2 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float16"} 0.000000e+00 : f16
    %3 = tensor.splat %2 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float16"} : tensor<2x16x1xf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<2x16x16xf16>, tensor<2x16x1xf16>) outs(%3 : tensor<2x16x1xf16>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float16"} {
    ^bb0(%5: f16, %6: f16, %7: f16):
      %8 = arith.mulf %5, %6 : f16
      %9 = arith.addf %7, %8 : f16
      linalg.yield %9 : f16
    } -> tensor<2x16x1xf16>
    func.return %4 : tensor<2x16x1xf16>
  }
}
