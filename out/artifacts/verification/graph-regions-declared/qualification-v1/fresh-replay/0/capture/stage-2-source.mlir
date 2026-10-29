"builtin.module"() ({
  "func.func"() <{sym_name = "source", function_type = (i8, i8) -> i8}> ({
  ^bb0(%0: i8, %1: i8):
    %r = "arith.muli"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()