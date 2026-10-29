"builtin.module"() ({
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 3 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %z = "arith.muli"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "scf.for"(%lb, %ub, %st, %a) ({
    ^bb0(%i: index, %c: i8):
      %v = "arith.addi"(%c, %z) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %o = "func.call"(%v) <{callee = @observe}> : (i8) -> i8
      "scf.yield"(%o) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    %out = "func.call"(%r) <{callee = @observe}> : (i8) -> i8
    "func.return"(%out) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i8) -> i8}> ({
  ^bb0(%a: i8):
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
}) : () -> ()