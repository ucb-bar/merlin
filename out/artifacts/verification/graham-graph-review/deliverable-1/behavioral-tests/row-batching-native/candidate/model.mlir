"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>) -> (tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>)}> ({
  ^bb0(%lhs: tensor<3x1x5xi16>, %rhs: tensor<1x7x5xi16>, %state: tensor<3x1x5xi16>):
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %r0 = "tensor.extract_slice"(%rhs) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 7, 5>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x7x5xi16>) -> tensor<1x7x5xi16>
    %rf0 = "tensor.collapse_shape"(%r0) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<1x7x5xi16>) -> tensor<35xi16>
    %re0 = "tensor.expand_shape"(%rf0) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 7, 5>}> : (tensor<35xi16>) -> tensor<7x5xi16>
    %rt0 = "tensor.empty"() : () -> tensor<5x7xi16>
    %b0 = "linalg.transpose"(%re0, %rt0) <{permutation = array<i64: 1, 0>}> ({
    ^bb1(%ti: i16, %to: i16):
      "linalg.yield"(%ti) : (i16) -> ()
    }) : (tensor<7x5xi16>, tensor<5x7xi16>) -> tensor<5x7xi16>
    %z0 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %0 = "tensor.collapse_shape"(%lhs) <{reassociation = [[0 : i64, 1 : i64, 2 : i64]]}> : (tensor<3x1x5xi16>) -> tensor<15xi16>
    %1 = "tensor.expand_shape"(%0) <{reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 3, 5>}> : (tensor<15xi16>) -> tensor<3x5xi16>
    %2 = "tensor.splat"(%zero) : (i32) -> tensor<3x7xi32>
    %3 = "linalg.generic"(%1, %b0, %2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb2(%a: i16, %b: i16, %c: i32):
      %ax = "arith.extsi"(%a) : (i16) -> i32
      %bx = "arith.extsi"(%b) : (i16) -> i32
      %p = "arith.muli"(%ax, %bx) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v = "arith.addi"(%p, %c) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "linalg.yield"(%v) : (i32) -> ()
    }) : (tensor<3x5xi16>, tensor<5x7xi16>, tensor<3x7xi32>) -> tensor<3x7xi32>
    %c0 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 7>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x7xi32>) -> tensor<1x7xi32>
    %z1 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %c1 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 1, 0>, static_sizes = array<i64: 1, 7>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x7xi32>) -> tensor<1x7xi32>
    %z2 = "tensor.splat"(%zero) : (i32) -> tensor<1x7xi32>
    %c2 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 2, 0>, static_sizes = array<i64: 1, 7>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<3x7xi32>) -> tensor<1x7xi32>
    %state_empty = "tensor.empty"() : () -> tensor<3x1x5xi16>
    %next_state = "linalg.generic"(%lhs, %state, %state_empty) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb3(%x: i16, %s: i16, %unused: i16):
      %updated = "arith.addi"(%x, %s) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      "linalg.yield"(%updated) : (i16) -> ()
    }) : (tensor<3x1x5xi16>, tensor<3x1x5xi16>, tensor<3x1x5xi16>) -> tensor<3x1x5xi16>
    "func.return"(%c0, %c1, %c2, %z0, %z1, %z2, %lhs, %rhs, %next_state) : (tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<1x7xi32>, tensor<3x1x5xi16>, tensor<1x7x5xi16>, tensor<3x1x5xi16>) -> ()
  }) {llvm.emit_c_interface} : () -> ()
}) : () -> ()