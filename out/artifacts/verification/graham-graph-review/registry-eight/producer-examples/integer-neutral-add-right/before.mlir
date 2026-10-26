"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %c = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %r = "arith.addi"(%a, %c) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()