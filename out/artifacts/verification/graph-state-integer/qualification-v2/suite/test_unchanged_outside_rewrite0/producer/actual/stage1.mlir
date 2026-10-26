"builtin.module"() ({
  "func.func"() <{sym_name = "main", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    %z = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %v = "arith.addi"(%x, %z) <{overflowFlags = #arith.overflow<nsw>}> : (i8, i8) -> i8
    "func.return"(%v) : (i8) -> ()
  }) : () -> ()
}) : () -> ()