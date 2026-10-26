"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %first = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %duplicate = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "func.call"(%duplicate) <{callee = @helper}> : (i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "helper", function_type = (i8) -> i8}> ({
  ^bb0(%v: i8):
    "func.return"(%v) : (i8) -> ()
  }) : () -> ()
}) : () -> ()