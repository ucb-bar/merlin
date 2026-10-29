"builtin.module"() ({
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %y = "arith.muli"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %second = "func.call"(%y) <{callee = @other}> : (i8) -> i8
    %z = "arith.subi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %first = "func.call"(%x) <{callee = @observe}> : (i8) -> i8
    %r = "arith.addi"(%first, %second) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %out = "arith.addi"(%r, %z) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%out) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%a: i8):
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
}) : () -> ()