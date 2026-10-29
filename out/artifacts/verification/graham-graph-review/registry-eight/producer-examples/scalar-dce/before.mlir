"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %dead = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %dead2 = "arith.muli"(%dead, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
}) : () -> ()