"builtin.module"() ({
  "func.func"() <{sym_name = "subject", function_type = (i1, i1) -> bf16}> ({
  ^bb0(%b: i1, %other: i1):
    %r = "arith.sitofp"(%b) : (i1) -> bf16
    "func.return"(%r) : (bf16) -> ()
  }) : () -> ()
}) : () -> ()