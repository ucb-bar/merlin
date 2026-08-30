builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<128x256xbf16>, %1: tensor<256x128xbf16>, %2: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %3 = tensor.empty() : tensor<128x128xbf16>
    %4 = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%4 : bf16) outs(%3 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %6 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%0, %1 : tensor<128x256xbf16>, tensor<256x128xbf16>) outs(%5 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %7 = tensor.empty() : tensor<128x128xbf16>
    %8 = arith.constant 0.000000e+00 : bf16
    %9 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%8 : bf16) outs(%7 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    %10 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%6, %2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%9 : tensor<128x128xbf16>) -> tensor<128x128xbf16>
    func.return %10 : tensor<128x128xbf16>
  }
}
