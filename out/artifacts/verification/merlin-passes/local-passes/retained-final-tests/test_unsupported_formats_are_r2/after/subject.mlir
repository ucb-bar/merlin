"builtin.module"() ({
  "func.func"() <{sym_name = "subject", function_type = (i1, i1) -> f64}> ({
  ^bb0(%b: i1, %other: i1):
    %r = "arith.uitofp"(%b) {prov.transforms = "bool_sitofp_uitofp"} : (i1) -> f64
    "func.return"(%r) : (f64) -> ()
  }) : () -> ()
}) : () -> ()