builtin.module attributes {prov.weights_file = "weights.safetensors"} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.transpose ins(%arg1:tensor<32x32xf32>) outs(%1:tensor<32x32xf32>) permutation = [1, 0]
    %3 = linalg.fill ins(%0 : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.matmul ins(%arg0, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = linalg.transpose ins(%arg2:tensor<32x32xf32>) outs(%1:tensor<32x32xf32>) permutation = [1, 0]
    %6 = linalg.matmul ins(%arg0, %5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) {
    ^bb0(%arg15: f32, %arg16: f32):
      %8 = arith.constant 1.000000e+00 : f32
      %9 = arith.negf %arg15 : f32
      %10 = math.exp %9 : f32
      %11 = arith.addf %8, %10 : f32
      %12 = arith.divf %8, %11 : f32
      linalg.yield %12 : f32
    } -> tensor<32x32xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %7 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) {
    ^bb1(%arg12: f32, %arg13: f32, %arg14: f32):
      %14 = arith.mulf %arg12, %arg13 : f32
      linalg.yield %14 : f32
    } -> tensor<32x32xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) {
    ^bb2(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      linalg.yield %16 : f32
    } -> tensor<32x32xf32>
    %17 = linalg.transpose ins(%arg3:tensor<32x32xf32>) outs(%1:tensor<32x32xf32>) permutation = [1, 0]
    %18 = linalg.matmul ins(%15, %17 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    func.return %18 : tensor<32x32xf32>
  }
}
