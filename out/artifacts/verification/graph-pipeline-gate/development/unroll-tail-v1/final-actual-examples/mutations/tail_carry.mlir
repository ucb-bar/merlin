"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (index, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "altered", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 3 : index}> : () -> index
    %ub = "arith.constant"() <{value = 18 : index}> : () -> index
    %st = "arith.constant"() <{value = 3 : index}> : () -> index
    %r = "arith.constant"() <{value = 6 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 15 : index}> : () -> index
    %r_2, %r_3 = "scf.for"(%lb, %r_1, %r, %a, %b) ({
    ^bb0(%0: index, %1: i8, %2: i8):
      %next = "func.call"(%0, %1) <{callee = @step}> : (index, i8) -> i8
      %3 = "arith.constant"() <{value = 3 : index}> : () -> index
      %4 = "arith.addi"(%0, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %next_1 = "func.call"(%4, %next) <{callee = @step}> : (index, i8) -> i8
      "scf.yield"(%next_1, %2) : (i8, i8) -> ()
    }) : (index, index, index, i8, i8) -> (i8, i8)
    %r_4, %r_5 = "scf.for"(%r_1, %ub, %st, %a, %b) ({
    ^bb0(%5: index, %6: i8, %7: i8):
      %next_2 = "func.call"(%5, %6) <{callee = @step}> : (index, i8) -> i8
      "scf.yield"(%next_2, %7) : (i8, i8) -> ()
    }) : (index, index, index, i8, i8) -> (i8, i8)
    %seen = "func.call"(%r_4) <{callee = @observe}> : (i8) -> i8
    "func.return"(%seen) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()