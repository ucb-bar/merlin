"builtin.module"() ({
  "func.func"() <{sym_name = "subject", function_type = (i1) -> f32}> ({
  ^bb0(%b: i1):
    %r = "scf.if"(%b) ({
      %t = "arith.sitofp"(%b) : (i1) -> f32
      "scf.yield"(%t) : (f32) -> ()
    }, {
      %f = "arith.sitofp"(%b) : (i1) -> f32
      "scf.yield"(%f) : (f32) -> ()
    }) : (i1) -> f32
    "func.return"(%r) : (f32) -> ()
  }) : () -> ()
}) : () -> ()