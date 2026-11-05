builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x16xi8>) -> (tensor<16x16xi32>) {
    %_component_0_empty = tensor.empty() : tensor<16x16xi32>
    %y = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%A0 : tensor<16x16xi8>) outs(%_component_0_empty : tensor<16x16xi32>) attrs = {prov.region_id = "component_0", prov.op = "cast", prov.family = "cast"} {
      ^bb0(%_component_0_element: i8, %_component_0_old: i32):
        %_component_0_converted = arith.extsi %_component_0_element : i8 to i32
        linalg.yield %_component_0_converted : i32
    } -> tensor<16x16xi32>
    func.return %y : tensor<16x16xi32>
  }
}
