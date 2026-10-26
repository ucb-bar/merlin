"builtin.module"() ({
  "func.func"() <{sym_name = "main", function_type = (f32) -> f32}> ({
  ^bb0(%x: f32):
    %z = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %v = "arith.addf"(%x, %z) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "func.return"(%v) : (f32) -> ()
  }) : () -> ()
}) : () -> ()