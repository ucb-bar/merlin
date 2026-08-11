builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_somvrbvk/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xbf16>, %1: tensor<16x32xbf16>) -> tensor<16x16xbf16> {
    %2 = tensor.empty() : tensor<32x16xbf16>
    %3 = linalg.transpose ins(%1:tensor<16x32xbf16>) outs(%2:tensor<32x16xbf16>) permutation = [1, 0]
    %4 = tensor.empty() : tensor<16x16xbf16>
    %5 = arith.constant 0.000000e+00 : bf16
    %6 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%5 : bf16) outs(%4 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %7 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16", prov.transposed_b = "true"} ins(%0, %3 : tensor<16x32xbf16>, tensor<32x16xbf16>) outs(%6 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    func.return %7 : tensor<16x16xbf16>
  }
}
