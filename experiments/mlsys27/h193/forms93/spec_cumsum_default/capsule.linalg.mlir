builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x16xi8>) -> (tensor<16x16xi32>) {
    %_component_0_empty = tensor.empty() : tensor<16x16xi32>
    %_component_0_zero = arith.constant 0 : i32
    %_component_0_init = linalg.fill ins(%_component_0_zero : i32) outs(%_component_0_empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %scan = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A0 : tensor<16x16xi8>) outs(%_component_0_init : tensor<16x16xi32>) attrs = {prov.region_id = "component_0", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.family = "reduction"} {
      ^bb0(%_component_0_element: i8, %_component_0_old: i32):
        %_component_0_row = linalg.index 0 : index
        %_component_0_scan = linalg.index 2 : index
        %_component_0_keep = arith.cmpi ule, %_component_0_scan, %_component_0_row : index
        %_component_0_wide = arith.extsi %_component_0_element : i8 to i32
        %_component_0_masked = arith.select %_component_0_keep, %_component_0_wide, %_component_0_zero : i32
        %_component_0_sum = arith.addi %_component_0_old, %_component_0_masked : i32
        linalg.yield %_component_0_sum : i32
    } -> tensor<16x16xi32>
    func.return %scan : tensor<16x16xi32>
  }
}
