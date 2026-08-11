builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_27b3bsys/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x32xbf16>, %1: tensor<32x16xbf16>) -> tensor<16x16xbf16> {
    %2 = tensor.empty() : tensor<16x16xbf16>
    %3 = arith.constant 0.000000e+00 : bf16
    %4 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3 : bf16) outs(%2 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %5 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %1 : tensor<16x32xbf16>, tensor<32x16xbf16>) outs(%4 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    func.return %5 : tensor<16x16xbf16>
  }
}
