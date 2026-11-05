builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x32xi8>) -> (tensor<32x16xi32>) {
    %_component_0_flat = tensor.collapse_shape %A0 [[0, 1]] : tensor<16x32xi8> into tensor<512xi8>
    %r = tensor.expand_shape %_component_0_flat [[0, 1]] output_shape [32, 16] : tensor<512xi8> into tensor<32x16xi8>
    %_component_1_empty = tensor.empty() : tensor<32x16xi32>
    %y = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%r : tensor<32x16xi8>) outs(%_component_1_empty : tensor<32x16xi32>) attrs = {prov.region_id = "component_1", prov.op = "cast", prov.family = "cast"} {
      ^bb1(%_component_1_element: i8, %_component_1_old: i32):
        %_component_1_converted = arith.extsi %_component_1_element : i8 to i32
        linalg.yield %_component_1_converted : i32
    } -> tensor<32x16xi32>
    func.return %y : tensor<32x16xi32>
  }
}
