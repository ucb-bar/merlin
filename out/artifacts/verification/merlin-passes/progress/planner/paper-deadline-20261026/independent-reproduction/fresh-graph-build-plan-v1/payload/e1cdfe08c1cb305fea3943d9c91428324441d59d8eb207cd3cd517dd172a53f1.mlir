"builtin.module"() ({
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %z = "arith.muli"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "func.call"(%z) <{callee = @observe}> : (i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i8) -> i8}> ({
  ^bb0(%a: i8):
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
}) : () -> ()