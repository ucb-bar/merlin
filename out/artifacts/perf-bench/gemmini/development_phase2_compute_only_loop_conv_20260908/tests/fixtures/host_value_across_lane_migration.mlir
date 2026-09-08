builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a0: tensor<2x4xi8>, %b0: tensor<4x3xi8>, %a1: tensor<2x4xi8>, %b1: tensor<4x3xi8>) -> tensor<2x3xf32> {
    %zero = arith.constant 0 : i8
    %e0 = tensor.empty() : tensor<2x3xi8>
    %z0 = linalg.fill {prov.region_id = "matmul_0", prov.family = "fill"} ins(%zero : i8) outs(%e0 : tensor<2x3xi8>) -> tensor<2x3xi8>
    %m0 = linalg.matmul {prov.region_id = "matmul_0", prov.family = "contraction"} ins(%a0, %b0 : tensor<2x4xi8>, tensor<4x3xi8>) outs(%z0 : tensor<2x3xi8>) -> tensor<2x3xi8>
    %he0 = tensor.empty() : tensor<2x3xf32>
    %h0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%m0 : tensor<2x3xi8>) outs(%he0 : tensor<2x3xf32>) attrs = {prov.region_id = "host_cast_0", prov.family = "cast"} {
    ^bb0(%x: i8, %unused: f32):
      %xf = arith.sitofp %x : i8 to f32
      linalg.yield %xf : f32
    } -> tensor<2x3xf32>
    %e1 = tensor.empty() : tensor<2x3xi8>
    %z1 = linalg.fill {prov.region_id = "matmul_1", prov.family = "fill"} ins(%zero : i8) outs(%e1 : tensor<2x3xi8>) -> tensor<2x3xi8>
    %m1 = linalg.matmul {prov.region_id = "matmul_1", prov.family = "contraction"} ins(%a1, %b1 : tensor<2x4xi8>, tensor<4x3xi8>) outs(%z1 : tensor<2x3xi8>) -> tensor<2x3xi8>
    %he1 = tensor.empty() : tensor<2x3xf32>
    %h1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%m1 : tensor<2x3xi8>) outs(%he1 : tensor<2x3xf32>) attrs = {prov.region_id = "host_cast_1", prov.family = "cast"} {
    ^bb1(%x: i8, %unused: f32):
      %xf = arith.sitofp %x : i8 to f32
      linalg.yield %xf : f32
    } -> tensor<2x3xf32>
    %sum_empty = tensor.empty() : tensor<2x3xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%h0, %h1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%sum_empty : tensor<2x3xf32>) attrs = {prov.region_id = "host_sum", prov.family = "elementwise"} {
    ^bb2(%x: f32, %y: f32, %unused: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<2x3xf32>
    func.return %sum : tensor<2x3xf32>
  }
}
