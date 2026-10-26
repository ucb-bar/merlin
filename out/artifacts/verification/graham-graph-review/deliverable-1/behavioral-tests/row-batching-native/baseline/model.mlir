"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>) -> (tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>)}> ({
  ^bb0(%lhs: tensor<3x1x5xi16>, %rhs: tensor<1x7x5xi16>, %state: tensor<3x1x5xi16>):
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %s0 = "tensor.extract_slice"(%lhs) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x1x5xi16>) -> tensor<1x1x5xi16>
    %f0 = "tensor.collapse_shape"(%s0) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x1x5xi16>) -> tensor<5xi16>
    %a0 = "tensor.expand_shape"(%f0) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 5>}> : (tensor<5xi16>) -> tensor<1x5xi16>
    %r0 = "tensor.extract_slice"(%rhs) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 7, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x7x5xi16>) -> tensor<1x7x5xi16>
    %rf0 = "tensor.collapse_shape"(%r0) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x7x5xi16>) -> tensor<35xi16>
    %re0 = "tensor.expand_shape"(%rf0) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 7, 5>}> : (tensor<35xi16>) -> tensor<7x5xi16>
    %rt0 = "tensor.empty"() : () -> tensor<5x7xi16>
    %b0 = "linalg.transpose"(%re0, %rt0) <{permutation = array<i64: 1, 0>}> ({
    ^bbt0(%ti: i16, %to: i16):
      "linalg.yield"(%ti) : (i16) -> ()
    }) : (tensor<7x5xi16>, tensor<5x7xi16>) -> tensor<5x7xi16>
    %z0 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %c0 = "linalg.generic"(%a0, %b0, %z0) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%a: i16, %b: i16, %c: i32):
      %ax = "arith.extsi"(%a) : (i16) -> i32
      %bx = "arith.extsi"(%b) : (i16) -> i32
      %p = "arith.muli"(%ax, %bx) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v = "arith.addi"(%p, %c) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%v) : (i32) -> ()
    }) : (tensor<1x5xi16>, tensor<5x7xi16>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %s1 = "tensor.extract_slice"(%lhs) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 1, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x1x5xi16>) -> tensor<1x1x5xi16>
    %f1 = "tensor.collapse_shape"(%s1) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x1x5xi16>) -> tensor<5xi16>
    %a1 = "tensor.expand_shape"(%f1) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 5>}> : (tensor<5xi16>) -> tensor<1x5xi16>
    %r1 = "tensor.extract_slice"(%rhs) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 7, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x7x5xi16>) -> tensor<1x7x5xi16>
    %rf1 = "tensor.collapse_shape"(%r1) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x7x5xi16>) -> tensor<35xi16>
    %re1 = "tensor.expand_shape"(%rf1) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 7, 5>}> : (tensor<35xi16>) -> tensor<7x5xi16>
    %rt1 = "tensor.empty"() : () -> tensor<5x7xi16>
    %b1 = "linalg.transpose"(%re1, %rt1) <{permutation = array<i64: 1, 0>}> ({
    ^bbt1(%ti_1: i16, %to_1: i16):
      "linalg.yield"(%ti_1) : (i16) -> ()
    }) : (tensor<7x5xi16>, tensor<5x7xi16>) -> tensor<5x7xi16>
    %z1 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %c1 = "linalg.generic"(%a1, %b1, %z1) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb2(%a_1: i16, %b_1: i16, %c_1: i32):
      %ax_1 = "arith.extsi"(%a_1) : (i16) -> i32
      %bx_1 = "arith.extsi"(%b_1) : (i16) -> i32
      %p_1 = "arith.muli"(%ax_1, %bx_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v_1 = "arith.addi"(%p_1, %c_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%v_1) : (i32) -> ()
    }) : (tensor<1x5xi16>, tensor<5x7xi16>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %s2 = "tensor.extract_slice"(%lhs) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 1, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x1x5xi16>) -> tensor<1x1x5xi16>
    %f2 = "tensor.collapse_shape"(%s2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x1x5xi16>) -> tensor<5xi16>
    %a2 = "tensor.expand_shape"(%f2) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 1, 5>}> : (tensor<5xi16>) -> tensor<1x5xi16>
    %r2 = "tensor.extract_slice"(%rhs) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 7, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x7x5xi16>) -> tensor<1x7x5xi16>
    %rf2 = "tensor.collapse_shape"(%r2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x7x5xi16>) -> tensor<35xi16>
    %re2 = "tensor.expand_shape"(%rf2) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 7, 5>}> : (tensor<35xi16>) -> tensor<7x5xi16>
    %rt2 = "tensor.empty"() : () -> tensor<5x7xi16>
    %b2 = "linalg.transpose"(%re2, %rt2) <{permutation = array<i64: 1, 0>}> ({
    ^bbt2(%ti_2: i16, %to_2: i16):
      "linalg.yield"(%ti_2) : (i16) -> ()
    }) : (tensor<7x5xi16>, tensor<5x7xi16>) -> tensor<5x7xi16>
    %z2 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %c2 = "linalg.generic"(%a2, %b2, %z2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb3(%a_2: i16, %b_2: i16, %c_2: i32):
      %ax_2 = "arith.extsi"(%a_2) : (i16) -> i32
      %bx_2 = "arith.extsi"(%b_2) : (i16) -> i32
      %p_2 = "arith.muli"(%ax_2, %bx_2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v_2 = "arith.addi"(%p_2, %c_2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%v_2) : (i32) -> ()
    }) : (tensor<1x5xi16>, tensor<5x7xi16>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %state_empty = "tensor.empty"() : () -> tensor<3x1x5xi16>
    %next_state = "linalg.generic"(%lhs, %state, %state_empty) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bbs(%x: i16, %s: i16, %unused: i16):
      %updated = "arith.addi"(%x, %s) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%updated) : (i16) -> ()
    }) : (tensor<3x1x5xi16>, tensor<3x1x5xi16>, tensor<3x1x5xi16>) -> tensor<3x1x5xi16>
    "func.return"(%c0, %c1, %c2, %z0, %z1, %z2, %lhs, %rhs, %next_state) : (tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
}) : () -> ()