builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x16xf32>, %1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %2 = tensor.empty() : tensor<16x16xf32>
    %3 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%0, %1 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%4 : tensor<16x16xf32>) -> tensor<16x16xf32>
    func.return %5 : tensor<16x16xf32>
  }
}
