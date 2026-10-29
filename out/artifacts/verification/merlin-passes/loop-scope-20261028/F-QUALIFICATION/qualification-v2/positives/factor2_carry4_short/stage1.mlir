"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> (i32, i32, i32, i32)}> ({
  ^bb0(%x: i32, %y: i32):
    %lb = "arith.constant"() <{value = 1 : index}> : () -> index
    %ub = "arith.constant"() <{value = 11 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %r = "arith.constant"() <{value = 2 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 11 : index}> : () -> index
    %r_2, %r_3, %r_4, %r_5 = "scf.for"(%lb, %r_1, %r, %x, %y, %x, %y) ({
    ^bb0(%0: index, %1: i32, %2: i32, %3: i32, %4: i32):
      %a0 = "arith.addi"(%1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b0 = "func.call"(%a0, %0) <{callee = @step}> : (i32, index) -> i32
      %a1 = "arith.addi"(%2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b1 = "func.call"(%a1, %0) <{callee = @step}> : (i32, index) -> i32
      %a2 = "arith.addi"(%3, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b2 = "func.call"(%a2, %0) <{callee = @step}> : (i32, index) -> i32
      %a3 = "arith.addi"(%4, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b3 = "func.call"(%a3, %0) <{callee = @step}> : (i32, index) -> i32
      "func.call"(%1, %y) <{callee = @observe}> : (i32, i32) -> ()
      "func.call"(%y, %1) <{callee = @observe}> : (i32, i32) -> ()
      %5 = "arith.constant"() <{value = 1 : index}> : () -> index
      %6 = "arith.addi"(%0, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a0_1 = "arith.addi"(%b0, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b0_1 = "func.call"(%a0_1, %6) <{callee = @step}> : (i32, index) -> i32
      %a1_1 = "arith.addi"(%b1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b1_1 = "func.call"(%a1_1, %6) <{callee = @step}> : (i32, index) -> i32
      %a2_1 = "arith.addi"(%b2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b2_1 = "func.call"(%a2_1, %6) <{callee = @step}> : (i32, index) -> i32
      %a3_1 = "arith.addi"(%b3, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b3_1 = "func.call"(%a3_1, %6) <{callee = @step}> : (i32, index) -> i32
      "func.call"(%b0, %y) <{callee = @observe}> : (i32, i32) -> ()
      "func.call"(%y, %b0) <{callee = @observe}> : (i32, i32) -> ()
      "scf.yield"(%b0_1, %b1_1, %b2_1, %b3_1) : (i32, i32, i32, i32) -> ()
    }) : (index, index, index, i32, i32, i32, i32) -> (i32, i32, i32, i32)
    "func.call"(%r_2) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_2, %r_3, %r_4, %r_5) : (i32, i32, i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()