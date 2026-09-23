builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%A0: tensor<16x32xi8>, %W0: tensor<32x16xi8>, %W1: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %acc0_empty = tensor.empty() : tensor<16x16xi32>
    %zero_acc0 = arith.constant 0 : i32
    %acc0_init = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%zero_acc0 : i32) outs(%acc0_empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %acc0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A0, %W0 : tensor<16x32xi8>, tensor<32x16xi8>) outs(%acc0_init : tensor<16x16xi32>) attrs = {prov.region_id = "contraction_0", prov.op = "matmul", prov.family = "contraction"} {
    ^bb_acc0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs_wide = arith.extsi %lhs : i8 to i32
      %rhs_wide = arith.extsi %rhs : i8 to i32
      %product = arith.muli %lhs_wide, %rhs_wide : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<16x16xi32>
    %narrow_empty = tensor.empty() : tensor<16x16xi8>
    %lo = arith.constant -128 : i32
    %hi = arith.constant 127 : i32
    %narrow = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc0 : tensor<16x16xi32>) outs(%narrow_empty : tensor<16x16xi8>) attrs = {prov.region_id = "requant_0", prov.op = "requant", prov.family = "elementwise_map", prov.placement = "accelerator_epilogue"} {
    ^bb0(%x: i32, %unused: i8):
      %clamp_lo = arith.maxsi %x, %lo : i32
      %clamp_hi = arith.minsi %clamp_lo, %hi : i32
      %narrowed = arith.trunci %clamp_hi : i32 to i8
      linalg.yield %narrowed : i8
    } -> tensor<16x16xi8>
    %acc1_empty = tensor.empty() : tensor<16x16xi32>
    %zero_acc1 = arith.constant 0 : i32
    %acc1_init = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%zero_acc1 : i32) outs(%acc1_empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%narrow, %W1 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc1_init : tensor<16x16xi32>) attrs = {prov.region_id = "contraction_1", prov.op = "matmul", prov.family = "contraction"} {
    ^bb_acc1(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs_wide = arith.extsi %lhs : i8 to i32
      %rhs_wide = arith.extsi %rhs : i8 to i32
      %product = arith.muli %lhs_wide, %rhs_wide : i32
      %sum = arith.addi %acc, %product : i32
      linalg.yield %sum : i32
    } -> tensor<16x16xi32>
    func.return %result : tensor<16x16xi32>
  }
}
