builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x16xi8>) -> (tensor<1x16xi32>) {
    %_component_0_empty = tensor.empty() : tensor<1x16xi32>
    %_component_0_reduced_empty = tensor.empty() : tensor<16xi32>
    %_component_0_zero = arith.constant 0 : i32
    %_component_0_init = linalg.fill ins(%_component_0_zero : i32) outs(%_component_0_reduced_empty : tensor<16xi32>) -> tensor<16xi32>
    %_component_0_reduced = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%A0 : tensor<16x16xi8>) outs(%_component_0_init : tensor<16xi32>) attrs = {prov.region_id = "component_0", prov.op = "reduce_sum", prov.family = "reduction"} {
      ^bb0(%_component_0_element: i8, %_component_0_old: i32):
        %_component_0_converted = arith.extsi %_component_0_element : i8 to i32
        %_component_0_sum = arith.addi %_component_0_old, %_component_0_converted : i32
        linalg.yield %_component_0_sum : i32
    } -> tensor<16xi32>
    %total = tensor.expand_shape %_component_0_reduced [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    func.return %total : tensor<1x16xi32>
  }
}
