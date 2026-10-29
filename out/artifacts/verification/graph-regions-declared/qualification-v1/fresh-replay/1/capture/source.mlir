"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>) -> (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>)}> ({
  ^bb0(%w0: tensor<2x23xi16>, %w1: tensor<2x23xi16>, %input: tensor<2x23xi16>, %state: tensor<2x23xi16>, %position: tensor<i32>):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %e0 = "tensor.empty"() : () -> tensor<2x23xi16>
    %e1 = "tensor.empty"() : () -> tensor<2x23xi16>
    %e2 = "tensor.empty"() : () -> tensor<2x23xi16>
    %p = "linalg.generic"(%input, %w0, %state, %e0) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%x: i16, %w: i16, %s: i16, %unused: i16):
      %product = "arith.muli"(%x, %w) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %r = "arith.addi"(%product, %s) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%r) : (i16) -> ()
    }) : (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>) -> tensor<2x23xi16>
    %c0 = "linalg.generic"(%p, %w1, %e1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%v: i16, %w_1: i16, %unused_1: i16):
      %r_1 = "arith.addi"(%v, %w_1) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%r_1) : (i16) -> ()
    }) : (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>) -> tensor<2x23xi16>
    %c1 = "linalg.generic"(%p, %w1, %e2) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%v_1: i16, %w_2: i16, %unused_2: i16):
      %r_2 = "arith.muli"(%v_1, %w_2) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%r_2) : (i16) -> ()
    }) : (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>) -> tensor<2x23xi16>
    %se = "tensor.empty"() : () -> tensor<2x23xi16>
    %next = "linalg.generic"(%c0, %c1, %position, %se) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%a: i16, %b: i16, %posarg: i32, %unused_3: i16):
      %sum = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %narrow = "arith.trunci"(%posarg) : (i32) -> i16
      %r_3 = "arith.addi"(%sum, %narrow) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%r_3) : (i16) -> ()
    }) : (tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>, tensor<2x23xi16>) -> tensor<2x23xi16>
    %old = "tensor.extract"(%position) : (tensor<i32>) -> i32
    %advanced = "arith.addi"(%old, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %pe = "tensor.empty"() : () -> tensor<i32>
    %nextpos = "tensor.insert"(%advanced, %pe) : (i32, tensor<i32>) -> tensor<i32>
    "func.return"(%c0, %c1, %next, %nextpos, %input, %w0, %w1, %state, %position) : (tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<2x23xi16>, tensor<i32>) -> ()
  }) : () -> ()
}) : () -> ()