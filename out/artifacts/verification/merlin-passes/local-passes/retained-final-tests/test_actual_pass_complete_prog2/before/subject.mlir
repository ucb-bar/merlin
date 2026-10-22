"builtin.module"() ({
  "func.func"() <{sym_name = "cast", function_type = (i1) -> f32, sym_visibility = "private"}> ({
  ^bb0(%b: i1):
    %r = "arith.sitofp"(%b) : (i1) -> f32
    "func.return"(%r) : (f32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "subject", function_type = (tensor<0xi1>) -> tensor<0xf32>}> ({
  ^bb0(%mask: tensor<0xi1>):
    %e = "tensor.empty"() : () -> tensor<0xf32>
    %r = "linalg.generic"(%mask, %e) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%b: i1, %old: f32):
      %f = "func.call"(%b) <{callee = @cast}> : (i1) -> f32
      "linalg.yield"(%f) : (f32) -> ()
    }) : (tensor<0xi1>, tensor<0xf32>) -> tensor<0xf32>
    "func.return"(%r) : (tensor<0xf32>) -> ()
  }) : () -> ()
}) : () -> ()