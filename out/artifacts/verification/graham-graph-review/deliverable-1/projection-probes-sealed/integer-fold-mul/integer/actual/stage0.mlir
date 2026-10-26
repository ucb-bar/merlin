"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %x = "arith.constant"() <{value = 127 : i8}> : () -> i8
    %y = "arith.constant"() <{value = 3 : i8}> : () -> i8
    %r = "arith.muli"(%x, %y) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()