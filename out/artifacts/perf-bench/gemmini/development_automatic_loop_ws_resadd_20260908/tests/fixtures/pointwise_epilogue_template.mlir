builtin.module attributes {prov.weights_file = "probe.weights", prov.level = "linalg-on-tensors", prov.quantization = "int8"} {
  func.func @forward(%a: tensor<@M@x@K@xi8>, %b: tensor<@K@x@N@xi8>, %scale: tensor<f32>, %channel: tensor<@N@xf32>, %bias: tensor<@N@xf32>, %residual: tensor<@M@x@N@xf32>, %qscale: tensor<f32>, %zero_point: tensor<i64>) -> tensor<@M@x@N@xi8> {
    %acc_empty = tensor.empty() : tensor<@M@x@N@xi32>
    %zero_i32 = arith.constant 0 : i32
    %init = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%zero_i32 : i32) outs(%acc_empty : tensor<@M@x@N@xi32>) -> tensor<@M@x@N@xi32>
    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<@M@x@K@xi8>, tensor<@K@x@N@xi8>) outs(%init : tensor<@M@x@N@xi32>) attrs = {prov.region_id = "contract", prov.family = "contraction", prov.op = "matmul", prov.orig_dtype = "int8"} {
    ^bb_contract(%lhs: i8, %rhs: i8, %sum: i32):
      %lhs_wide = arith.extsi %lhs : i8 to i32
      %rhs_wide = arith.extsi %rhs : i8 to i32
      %product = arith.muli %lhs_wide, %rhs_wide : i32
      %updated = arith.addi %sum, %product : i32
      linalg.yield %updated : i32
    } -> tensor<@M@x@N@xi32>
    %affine_empty = tensor.empty() : tensor<@M@x@N@xf32>
    %affine = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], prov.region_id = "affine_bias_residual_relu", prov.family = "elementwise", prov.op = "affine_bias_residual_relu", prov.orig_dtype = "float32"} ins(%acc, %scale, %channel, %bias, %residual : tensor<@M@x@N@xi32>, tensor<f32>, tensor<@N@xf32>, tensor<@N@xf32>, tensor<@M@x@N@xf32>) outs(%affine_empty : tensor<@M@x@N@xf32>) {
    ^bb0(%x: i32, %s: f32, %c: f32, %b0: f32, %r: f32, %unused: f32):
      %xf = arith.sitofp %x : i32 to f32
      %scaled = arith.mulf %xf, %s : f32
      %per_channel = arith.mulf %scaled, %c : f32
      %biased = arith.addf %per_channel, %b0 : f32
      %with_residual = arith.addf %biased, %r : f32
      %zero_f32 = arith.constant 0.000000e+00 : f32
      %activated = arith.maximumf %with_residual, %zero_f32 : f32
      linalg.yield %activated : f32
    } -> tensor<@M@x@N@xf32>
    %out_empty = tensor.empty() : tensor<@M@x@N@xi8>
    %lo = arith.constant -1.280000e+02 : f32
    %hi = arith.constant 1.270000e+02 : f32
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], prov.region_id = "quantize", prov.family = "quantize", prov.op = "quantize_per_tensor", prov.orig_dtype = "int8"} ins(%affine, %qscale, %zero_point : tensor<@M@x@N@xf32>, tensor<f32>, tensor<i64>) outs(%out_empty : tensor<@M@x@N@xi8>) {
    ^bb1(%x: f32, %qs: f32, %zp: i64, %unused: i8):
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
