builtin.module attributes {prov.weights_file = "probe.weights", prov.level = "linalg-on-tensors", prov.quantization = "int8"} {
  func.func @forward(%a: tensor<@M@x@N@xi8>, %b: tensor<@M@x@N@xi8>) -> tensor<@M@x@N@xi8> {
    %empty = tensor.empty() : tensor<@M@x@N@xi8>
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<@M@x@N@xi8>, tensor<@M@x@N@xi8>) outs(%empty : tensor<@M@x@N@xi8>) attrs = {prov.region_id = "resadd", prov.family = "elementwise", prov.op = "saturating_add", prov.orig_dtype = "int8"} {
    ^bb0(%x: i8, %y: i8, %unused: i8):
      %x32 = arith.extsi %x : i8 to i32
      %y32 = arith.extsi %y : i8 to i32
      %sum = arith.addi %x32, %y32 : i32
      %lo = arith.constant @LO@ : i32
      %clamped_lo = arith.maxsi %sum, %lo : i32
      %hi = arith.constant 127 : i32
      %clamped = arith.minsi %clamped_lo, %hi : i32
      %narrow = arith.trunci %clamped : i32 to i8
      linalg.yield %narrow : i8
    } -> tensor<@M@x@N@xi8>
    return %out : tensor<@M@x@N@xi8>
  }
}
