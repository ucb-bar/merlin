builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %1 = tensor.empty() : tensor<32x32xbf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<32x32xbf16>) outs(%1 : tensor<32x32xbf16>) attrs =  {prov.region_id = "requant_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float8_e4m3fn"} {
    ^bb0(%3: bf16, %4: bf16):
      %5 = arith.extf %3 : bf16 to f32
      %6 = arith.truncf %5 : f32 to f8E4M3FN
      %7 = arith.extf %6 : f8E4M3FN to f32
      %8 = arith.truncf %7 : f32 to bf16
      linalg.yield %8 : bf16
    } -> tensor<32x32xbf16>
    func.return %2 : tensor<32x32xbf16>
  }
}
