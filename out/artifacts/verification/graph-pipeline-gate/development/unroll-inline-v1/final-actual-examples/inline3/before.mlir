"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (index, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "altered", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 7 : index}> : () -> index
    %ub = "arith.constant"() <{value = 16 : index}> : () -> index
    %st = "arith.constant"() <{value = 3 : index}> : () -> index
    %r, %r_1 = "scf.for"(%lb, %ub, %st, %a, %b) ({
    ^bb0(%i: index, %c: i8, %d: i8):
      %sum = "arith.addi"(%c, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %next = "func.call"(%i, %sum) <{callee = @step}> : (index, i8) -> i8
      "scf.yield"(%next, %d) : (i8, i8) -> ()
    }) : (index, index, index, i8, i8) -> (i8, i8)
    %seen = "func.call"(%r) <{callee = @observe}> : (i8) -> i8
    "func.return"(%seen) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()