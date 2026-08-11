builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m_35asqkdf/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf16>, %1: tensor<16x16xf16>, %2: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %3 = tensor.empty() : tensor<16x16xf16>
    %4 = arith.constant 0.000000e+00 : f16
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : f16) outs(%3 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float16"} ins(%0, %1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%5 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %7 = tensor.empty() : tensor<16x16xf16>
    %8 = arith.constant 0.000000e+00 : f16
    %9 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%8 : f16) outs(%7 : tensor<16x16xf16>) -> tensor<16x16xf16>
    %10 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float16"} ins(%6, %2 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%9 : tensor<16x16xf16>) -> tensor<16x16xf16>
    func.return %10 : tensor<16x16xf16>
  }
}
