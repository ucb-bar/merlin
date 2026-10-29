"builtin.module"() ({
  "func.func"() <{sym_name = "source", function_type = (i16, i16, i16) -> i16}> ({
  ^bb0(%0: i16, %1: i16, %2: i16):
    %product = "arith.muli"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %r = "arith.addi"(%product, %2) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    "func.return"(%r) : (i16) -> ()
  }) : () -> ()
}) : () -> ()