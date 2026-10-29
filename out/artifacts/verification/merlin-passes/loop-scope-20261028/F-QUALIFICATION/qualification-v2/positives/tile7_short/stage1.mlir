"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> i32}> ({
  ^bb0(%x: i32, %y: i32):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 27 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %r = "arith.constant"() <{value = 3 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 27 : index}> : () -> index
    %r_2 = "scf.for"(%lb, %r_1, %r, %x) ({
    ^bb0(%0: index, %1: i32):
      %a0 = "arith.addi"(%1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b0 = "func.call"(%a0, %0) <{callee = @step}> : (i32, index) -> i32
      "func.call"(%1, %y) <{callee = @observe}> : (i32, i32) -> ()
      "func.call"(%y, %1) <{callee = @observe}> : (i32, i32) -> ()
      %2 = "arith.constant"() <{value = 1 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a0_1 = "arith.addi"(%b0, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b0_1 = "func.call"(%a0_1, %3) <{callee = @step}> : (i32, index) -> i32
      "func.call"(%b0, %y) <{callee = @observe}> : (i32, i32) -> ()
      "func.call"(%y, %b0) <{callee = @observe}> : (i32, i32) -> ()
      %4 = "arith.constant"() <{value = 2 : index}> : () -> index
      %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a0_2 = "arith.addi"(%b0_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b0_2 = "func.call"(%a0_2, %5) <{callee = @step}> : (i32, index) -> i32
      "func.call"(%b0_1, %y) <{callee = @observe}> : (i32, i32) -> ()
      "func.call"(%y, %b0_1) <{callee = @observe}> : (i32, i32) -> ()
      "scf.yield"(%b0_2) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%r_2) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_2) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()