builtin.module attributes {prov.weights_file = "weights.safetensors", prov.level = "linalg-on-tensors"} {
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
      %23 = arith.constant 0 : i64
      %24 = arith.sitofp %23 : i64 to f32
      %25 = arith.cmpf olt, %21, %24 : f32
      %26 = arith.select %25, %24, %21 : f32
      linalg.yield %26 : f32
    } -> tensor<4x32xf32>
    %27 = tensor.empty() : tensor<32x32xf32>
    %28 = linalg.transpose ins(%2:tensor<32x32xf32>) outs(%27:tensor<32x32xf32>) permutation = [1, 0]
    %29 = tensor.empty() : tensor<4x32xf32>
    %30 = arith.constant {prov.module = "fc2"} 0.000000e+00 : f32
    %31 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc2"} ins(%30 : f32) outs(%29 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %32 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc2", prov.fqn = "fc2", prov.transposed_b = "true"} ins(%20, %28 : tensor<4x32xf32>, tensor<32x32xf32>) outs(%31 : tensor<4x32xf32>) -> tensor<4x32xf32>
    %33 = tensor.empty() : tensor<4x32xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %3 : tensor<4x32xf32>, tensor<32xf32>) outs(%33 : tensor<4x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc2", prov.fqn = "fc2"} {
    ^bb2(%35: f32, %36: f32, %37: f32):
      %38 = arith.addf %35, %36 : f32
      linalg.yield %38 : f32
    } -> tensor<4x32xf32>
    %39 = tensor.empty() : tensor<4x32xf32>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<4x32xf32>) outs(%39 : tensor<4x32xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "act2", prov.fqn = "act2"} {
    ^bb3(%41: f32, %42: f32):
      %43 = arith.constant 0 : i64
      %44 = arith.sitofp %43 : i64 to f32
      %45 = arith.cmpf olt, %41, %44 : f32
      %46 = arith.select %45, %44, %41 : f32
      linalg.yield %46 : f32
    } -> tensor<4x32xf32>
    %47 = tensor.empty() : tensor<32x8xf32>
    %48 = linalg.transpose ins(%4:tensor<8x32xf32>) outs(%47:tensor<32x8xf32>) permutation = [1, 0]
    %49 = tensor.empty() : tensor<4x8xf32>
    %50 = arith.constant {prov.module = "fc3"} 0.000000e+00 : f32
    %51 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "fc3"} ins(%50 : f32) outs(%49 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %52 = linalg.matmul {prov.region_id = "matmul_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3", prov.transposed_b = "true"} ins(%40, %48 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%51 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %53 = tensor.empty() : tensor<4x8xf32>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52, %5 : tensor<4x8xf32>, tensor<8xf32>) outs(%53 : tensor<4x8xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "fc3", prov.fqn = "fc3"} {
    ^bb4(%55: f32, %56: f32, %57: f32):
      %58 = arith.addf %55, %56 : f32
      linalg.yield %58 : f32
    } -> tensor<4x8xf32>
    func.return %54 : tensor<4x8xf32>
  }
}