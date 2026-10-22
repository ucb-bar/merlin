"builtin.module"() ({
  "func.func"() <{sym_name = "subject", function_type = (tensor<18xf32>) -> tensor<18xf32>}> ({
  ^bb0(%x: tensor<18xf32>):
    %e = "tensor.empty"() : () -> tensor<18xf32>
    %r = "linalg.generic"(%x, %e) <{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%v: f32, %old: f32):
      %zero = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %b = "arith.cmpf"(%v, %zero) <{predicate = 13 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
      %out = "arith.uitofp"(%b) {prov.transforms = "bool_sitofp_uitofp"} : (i1) -> f32
      "linalg.yield"(%out) : (f32) -> ()
    }) : (tensor<18xf32>, tensor<18xf32>) -> tensor<18xf32>
    "func.return"(%r) : (tensor<18xf32>) -> ()
  }) : () -> ()
}) : () -> ()