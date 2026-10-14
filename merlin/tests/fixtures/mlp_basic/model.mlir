builtin.module attributes {prov.weights_file = "workloads/mlp_basic/mlp_basic.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x16xf32>, %1: tensor<32xf32>, %2: tensor<32x32xf32>, %3: tensor<32xf32>, %4: tensor<8x32xf32>, %5: tensor<8xf32>, %6: tensor<4x16xf32>) -> tensor<4x8xf32> {
    %7 = tensor.empty() : tensor<16x32xf32>
    %8 = linalg.transpose ins(%0:tensor<32x16xf32>) outs(%7:tensor<16x32xf32>) permutation = [1, 0]
    %9 = tensor.empty() : tensor<4x32xf32>
    %10 = arith.constant {prov.module = "fc1"} 0.000000e+00 : f32
    %11 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc1"} ins(%10 : f32) outs(%9 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %12 = linalg.matmul {prov.region_id = "matmul_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc1", prov.fqn = "fc1", prov.transposed_b = "true"} ins(%6, %8 : tensor<4x16xf32>, tensor<16x32xf32>) outs(%11 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %13 = tensor.empty() : tensor<4x32xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %1 : tensor<4x32xf32>, tensor<32xf32>) outs(%13 : tensor<4x32xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc1", prov.fqn = "fc1"} {
    ^bb0(%15: f32, %16: f32, %17: f32):
      %18 = arith.addf %15, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<4x32xf32>
    %19 = tensor.empty() : tensor<4x32xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<4x32xf32>) outs(%19 : tensor<4x32xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "act1", prov.fqn = "act1"} {
    ^bb1(%21: f32, %22: f32):
      %23 = arith.constant 0.000000e+00 : f32
      %24 = arith.maximumf %21, %23 : f32
      linalg.yield %24 : f32
    } -> tensor<4x32xf32>
    %25 = tensor.empty() : tensor<32x32xf32>
    %26 = linalg.transpose ins(%2:tensor<32x32xf32>) outs(%25:tensor<32x32xf32>) permutation = [1, 0]
    %27 = tensor.empty() : tensor<4x32xf32>
    %28 = arith.constant {prov.module = "fc2"} 0.000000e+00 : f32
    %29 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc2"} ins(%28 : f32) outs(%27 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %30 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc2", prov.fqn = "fc2", prov.transposed_b = "true"} ins(%20, %26 : tensor<4x32xf32>, tensor<32x32xf32>) outs(%29 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %31 = tensor.empty() : tensor<4x32xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30, %3 : tensor<4x32xf32>, tensor<32xf32>) outs(%31 : tensor<4x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc2", prov.fqn = "fc2"} {
    ^bb2(%33: f32, %34: f32, %35: f32):
      %36 = arith.addf %33, %34 : f32
      linalg.yield %36 : f32
    } -> tensor<4x32xf32>
    %37 = tensor.empty() : tensor<4x32xf32>
    %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<4x32xf32>) outs(%37 : tensor<4x32xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "act2", prov.fqn = "act2"} {
    ^bb3(%39: f32, %40: f32):
      %41 = arith.constant 0.000000e+00 : f32
      %42 = arith.maximumf %39, %41 : f32
      linalg.yield %42 : f32
    } -> tensor<4x32xf32>
    %43 = tensor.empty() : tensor<32x8xf32>
    %44 = linalg.transpose ins(%4:tensor<8x32xf32>) outs(%43:tensor<32x8xf32>) permutation = [1, 0]
    %45 = tensor.empty() : tensor<4x8xf32>
    %46 = arith.constant {prov.module = "fc3"} 0.000000e+00 : f32
    %47 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc3"} ins(%46 : f32) outs(%45 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %48 = linalg.matmul {prov.region_id = "matmul_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3", prov.transposed_b = "true"} ins(%38, %44 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%47 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %49 = tensor.empty() : tensor<4x8xf32>
    %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48, %5 : tensor<4x8xf32>, tensor<8xf32>) outs(%49 : tensor<4x8xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3"} {
    ^bb4(%51: f32, %52: f32, %53: f32):
      %54 = arith.addf %51, %52 : f32
      linalg.yield %54 : f32
    } -> tensor<4x8xf32>
    func.return %50 : tensor<4x8xf32>
  }
}
