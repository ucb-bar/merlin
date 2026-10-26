"builtin.module"() ({
  "func.func"() <{sym_name = "callee", function_type = (i8) -> i8}> ({
  ^bb0(%v: i8):
    %w = "arith.subi"(%v, %v) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%w) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "k", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %f = "func.constant"() <{value = @callee}> : () -> ((i8) -> i8)
    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %z = "arith.muli"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "func.call_indirect"(%f, %z) : ((i8) -> i8, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()