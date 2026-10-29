"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %y = "arith.muli"(%b, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %x = "arith.addi"(%a, %a) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "arith.subi"(%x, %y) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()