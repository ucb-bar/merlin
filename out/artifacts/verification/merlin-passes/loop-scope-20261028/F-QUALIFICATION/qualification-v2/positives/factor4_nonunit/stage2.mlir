"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> (i32, i32)}> ({
  ^bb0(%x: i32, %y: i32):
    %lb = "arith.constant"() <{value = 9 : index}> : () -> index
    %ub = "arith.constant"() <{value = 57 : index}> : () -> index
    %st = "arith.constant"() <{value = 4 : index}> : () -> index
    %r = "arith.constant"() <{value = 16 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 57 : index}> : () -> index
    %r_2 = "arith.constant"() <{value = 32 : index}> : () -> index
    %r_3, %r_4 = "scf.for"(%lb, %r_1, %r_2, %x, %y) ({
    ^bb0(%0: index, %1: i32, %2: i32):
      %3 = "arith.addi"(%0, %r_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %4 = "arith.minui"(%3, %r_1) : (index, index) -> index
      %5, %6 = "scf.for"(%0, %4, %r, %1, %2) ({
      ^bb0(%7: index, %8: i32, %9: i32):
        %a0 = "arith.addi"(%8, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0 = "func.call"(%a0, %7) <{callee = @step}> : (i32, index) -> i32
        %a1 = "arith.addi"(%9, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1 = "func.call"(%a1, %7) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%8, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %8) <{callee = @observe}> : (i32, i32) -> ()
        %10 = "arith.constant"() <{value = 4 : index}> : () -> index
        %11 = "arith.addi"(%7, %10) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_1 = "arith.addi"(%b0, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_1 = "func.call"(%a0_1, %11) <{callee = @step}> : (i32, index) -> i32
        %a1_1 = "arith.addi"(%b1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_1 = "func.call"(%a1_1, %11) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %b0) <{callee = @observe}> : (i32, i32) -> ()
        %12 = "arith.constant"() <{value = 8 : index}> : () -> index
        %13 = "arith.addi"(%7, %12) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_2 = "arith.addi"(%b0_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_2 = "func.call"(%a0_2, %13) <{callee = @step}> : (i32, index) -> i32
        %a1_2 = "arith.addi"(%b1_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_2 = "func.call"(%a1_2, %13) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0_1, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %b0_1) <{callee = @observe}> : (i32, i32) -> ()
        %14 = "arith.constant"() <{value = 12 : index}> : () -> index
        %15 = "arith.addi"(%7, %14) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_3 = "arith.addi"(%b0_2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_3 = "func.call"(%a0_3, %15) <{callee = @step}> : (i32, index) -> i32
        %a1_3 = "arith.addi"(%b1_2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_3 = "func.call"(%a1_3, %15) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0_2, %y) <{callee = @observe}> : (i32, i32) -> ()
        "func.call"(%y, %b0_2) <{callee = @observe}> : (i32, i32) -> ()
        "scf.yield"(%b0_3, %b1_3) : (i32, i32) -> ()
      }) : (index, index, index, i32, i32) -> (i32, i32)
      "scf.yield"(%5, %6) : (i32, i32) -> ()
    }) : (index, index, index, i32, i32) -> (i32, i32)
    "func.call"(%r_3) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_3, %r_4) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()