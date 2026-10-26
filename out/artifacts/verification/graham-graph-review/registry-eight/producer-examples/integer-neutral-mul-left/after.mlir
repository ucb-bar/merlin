"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %c = "arith.constant"() <{value = 1 : i8}> : () -> i8
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
}) : () -> ()