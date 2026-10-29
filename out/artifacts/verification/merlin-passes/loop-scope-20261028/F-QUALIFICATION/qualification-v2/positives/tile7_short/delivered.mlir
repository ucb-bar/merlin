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
    %r_2 = "arith.constant"() <{value = 21 : index}> : () -> index
    %r_3 = "scf.for"(%lb, %r_1, %r_2, %x) ({
    ^bb0(%0: index, %1: i32):
      %2 = "arith.addi"(%0, %r_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %r_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %r, %1) ({
      ^bb0(%5: index, %6: i32):
        %a0 = "arith.addi"(%6, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0 = "func.call"(%a0, %5) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%6, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %6) <{callee = @observe}> : (i32, i32) -> ()
        %7 = "arith.constant"() <{value = 1 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_1 = "arith.addi"(%b0, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_1 = "func.call"(%a0_1, %8) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %b0) <{callee = @observe}> : (i32, i32) -> ()
        %9 = "arith.constant"() <{value = 2 : index}> : () -> index
        %10 = "arith.addi"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_2 = "arith.addi"(%b0_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_2 = "func.call"(%a0_2, %10) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0_1, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %b0_1) <{callee = @observe}> : (i32, i32) -> ()
        "scf.yield"(%b0_2) : (i32) -> ()
      }) : (index, index, index, i32) -> i32
      "scf.yield"(%4) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%r_3) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_3) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()