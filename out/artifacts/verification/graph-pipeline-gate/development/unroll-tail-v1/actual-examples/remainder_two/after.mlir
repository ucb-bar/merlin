"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (index, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "altered", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 1 : index}> : () -> index
    %ub = "arith.constant"() <{value = 17 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 6 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 13 : index}> : () -> index
    %r_2 = "scf.for"(%lb, %r_1, %r, %a) ({
    ^bb0(%0: index, %1: i8):
      %next = "func.call"(%0, %1) <{callee = @step}> : (index, i8) -> i8
      %2 = "arith.constant"() <{value = 2 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %next_1 = "func.call"(%3, %next) <{callee = @step}> : (index, i8) -> i8
      %4 = "arith.constant"() <{value = 4 : index}> : () -> index
      %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %next_2 = "func.call"(%5, %next_1) <{callee = @step}> : (index, i8) -> i8
      "scf.yield"(%next_2) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    %r_3 = "scf.for"(%r_1, %ub, %st, %r_2) ({
    ^bb0(%6: index, %7: i8):
      %next_3 = "func.call"(%6, %7) <{callee = @step}> : (index, i8) -> i8
      "scf.yield"(%next_3) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    %seen = "func.call"(%r_3) <{callee = @observe}> : (i8) -> i8
    "func.return"(%seen) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()