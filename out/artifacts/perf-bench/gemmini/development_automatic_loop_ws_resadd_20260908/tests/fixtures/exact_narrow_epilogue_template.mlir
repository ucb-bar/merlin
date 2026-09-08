builtin.module attributes {prov.weights_file = "probe.weights", prov.level = "linalg-on-tensors", prov.quantization = "int8"} {
  func.func @forward(%a: tensor<@M@x@K@xi8>, %b: tensor<@K@x@N@xi8>, %acc_scale: tensor<f32>, %channel_scale: tensor<@N@xf32>, %bias: tensor<@N@xf32>, %qscale: tensor<f32>) -> tensor<@M@x@N@xi8> {
    %acc_empty = tensor.empty() : tensor<@M@x@N@xi32>
    %zero_i32 = arith.constant 0 : i32
    %init = linalg.fill ins(%zero_i32 : i32) outs(%acc_empty : tensor<@M@x@N@xi32>) -> tensor<@M@x@N@xi32>
    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<@M@x@K@xi8>, tensor<@K@x@N@xi8>) outs(%init : tensor<@M@x@N@xi32>) attrs = {prov.region_id = "contract", prov.family = "contraction", prov.op = "matmul", prov.orig_dtype = "int8"} {
    ^bb0(%lhs: i8, %rhs: i8, %sum: i32):
      %lhs_wide = arith.extsi %lhs : i8 to i32
      %rhs_wide = arith.extsi %rhs : i8 to i32
      %product = arith.muli %lhs_wide, %rhs_wide : i32
      %updated = arith.addi %product, %sum : i32
      linalg.yield %updated : i32
    } -> tensor<@M@x@N@xi32>
    %affine_empty = tensor.empty() : tensor<@M@x@N@xf32>
    %affine = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc, %acc_scale, %channel_scale : tensor<@M@x@N@xi32>, tensor<f32>, tensor<@N@xf32>) outs(%affine_empty : tensor<@M@x@N@xf32>) attrs = {prov.region_id = "affine", prov.family = "elementwise", prov.op = "affine"} {
    ^bb1(%x: i32, %s: f32, %cs: f32, %unused: f32):
      %xf = arith.sitofp %x : i32 to f32
      %scaled = arith.mulf %xf, %s : f32
      %channel_scaled = arith.mulf %scaled, %cs : f32
      linalg.yield %channel_scaled : f32
    } -> tensor<@M@x@N@xf32>
    %bias_empty = tensor.empty() : tensor<@M@x@N@xf32>
    %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%affine, %bias : tensor<@M@x@N@xf32>, tensor<@N@xf32>) outs(%bias_empty : tensor<@M@x@N@xf32>) attrs = {prov.region_id = "bias", prov.family = "elementwise", prov.op = "bias_add"} {
    ^bb2(%x: f32, %b0: f32, %unused: f32):
      %sum = arith.addf %x, %b0 : f32
      linalg.yield %sum : f32
    } -> tensor<@M@x@N@xf32>
    %relu_empty = tensor.empty() : tensor<@M@x@N@xf32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%biased : tensor<@M@x@N@xf32>) outs(%relu_empty : tensor<@M@x@N@xf32>) attrs = {prov.region_id = "relu", prov.family = "minmax", prov.op = "relu"} {
    ^bb3(%x: f32, %unused: f32):
      %zero = arith.constant 0.000000e+00 : f32
      %activated = arith.maximumf %x, %zero : f32
      linalg.yield %activated : f32
    } -> tensor<@M@x@N@xf32>
    %reciprocal_empty = tensor.empty() : tensor<f32>
    %reciprocal = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%qscale : tensor<f32>) outs(%reciprocal_empty : tensor<f32>) attrs = {prov.region_id = "qscale", prov.family = "quantize", prov.op = "reciprocal"} {
    ^bb4(%x: f32, %unused: f32):
      %one = arith.constant 1.000000e+00 : f32
      %r = arith.divf %one, %x : f32
      linalg.yield %r : f32
    } -> tensor<f32>
    %out_empty = tensor.empty() : tensor<@M@x@N@xi8>
    %zero_point_scalar = arith.constant 0 : i64
    %zero_point = tensor.splat %zero_point_scalar : tensor<i64>
    %lo = arith.constant -1.280000e+02 : f32
    %hi = arith.constant 1.270000e+02 : f32
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%relu, %reciprocal, %zero_point : tensor<@M@x@N@xf32>, tensor<f32>, tensor<i64>) outs(%out_empty : tensor<@M@x@N@xi8>) attrs = {prov.region_id = "quantize", prov.family = "quantize", prov.op = "quantize_per_tensor"} {
    ^bb5(%x: f32, %qs: f32, %zp: i64, %unused: i8):
      %scaled = arith.mulf %x, %qs : f32
      %rounded = math.roundeven %scaled : f32
      %zpf = arith.sitofp %zp : i64 to f32
      %shifted = arith.addf %rounded, %zpf : f32
      %clamped_lo = arith.maximumf %shifted, %lo : f32
      %clamped = arith.minimumf %clamped_lo, %hi : f32
      %narrow = arith.fptosi %clamped : f32 to i8
      linalg.yield %narrow : i8
    } -> tensor<@M@x@N@xi8>
    return %out : tensor<@M@x@N@xi8>
  }
}
