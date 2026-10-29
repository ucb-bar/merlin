"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> (i32, i32, i32)}> ({
  ^bb0(%x: i32, %y: i32):
    %lb = "arith.constant"() <{value = 2 : index}> : () -> index
    %ub = "arith.constant"() <{value = 42 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 8 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 42 : index}> : () -> index
    %r_2 = "arith.constant"() <{value = 32 : index}> : () -> index
    %r_3, %r_4, %r_5 = "scf.for"(%lb, %r_1, %r_2, %x, %y, %x) ({
    ^bb0(%0: index, %1: i32, %2: i32, %3: i32):
      %4 = "arith.addi"(%0, %r_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %5 = "arith.minui"(%4, %r_1) : (index, index) -> index
      %6, %7, %8 = "scf.for"(%0, %5, %r, %1, %2, %3) ({
      ^bb0(%9: index, %10: i32, %11: i32, %12: i32):
        %a0 = "arith.addi"(%10, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0 = "func.call"(%a0, %9) <{callee = @step}> : (i32, index) -> i32
        %a1 = "arith.addi"(%11, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1 = "func.call"(%a1, %9) <{callee = @step}> : (i32, index) -> i32
        %a2 = "arith.addi"(%12, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b2 = "func.call"(%a2, %9) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%10, %y) <{callee = @observe}> : (i32, i32) -> ()
        %13 = "arith.constant"() <{value = 2 : index}> : () -> index
        %14 = "arith.addi"(%9, %13) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_1 = "arith.addi"(%b0, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_1 = "func.call"(%a0_1, %14) <{callee = @step}> : (i32, index) -> i32
        %a1_1 = "arith.addi"(%b1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_1 = "func.call"(%a1_1, %14) <{callee = @step}> : (i32, index) -> i32
        %a2_1 = "arith.addi"(%b2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b2_1 = "func.call"(%a2_1, %14) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0, %y) <{callee = @observe}> : (i32, i32) -> ()
        %15 = "arith.constant"() <{value = 4 : index}> : () -> index
        %16 = "arith.addi"(%9, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_2 = "arith.addi"(%b0_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_2 = "func.call"(%a0_2, %16) <{callee = @step}> : (i32, index) -> i32
        %a1_2 = "arith.addi"(%b1_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_2 = "func.call"(%a1_2, %16) <{callee = @step}> : (i32, index) -> i32
        %a2_2 = "arith.addi"(%b2_1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b2_2 = "func.call"(%a2_2, %16) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0_1, %y) <{callee = @observe}> : (i32, i32) -> ()
        %17 = "arith.constant"() <{value = 6 : index}> : () -> index
        %18 = "arith.addi"(%9, %17) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a0_3 = "arith.addi"(%b0_2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b0_3 = "func.call"(%a0_3, %18) <{callee = @step}> : (i32, index) -> i32
        %a1_3 = "arith.addi"(%b1_2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b1_3 = "func.call"(%a1_3, %18) <{callee = @step}> : (i32, index) -> i32
        %a2_3 = "arith.addi"(%b2_2, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b2_3 = "func.call"(%a2_3, %18) <{callee = @step}> : (i32, index) -> i32
        "func.call"(%b0_2, %y) <{callee = @observe}> : (i32, i32) -> ()
        "scf.yield"(%b0_3, %b1_3, %b2_3) : (i32, i32, i32) -> ()
      }) : (index, index, index, i32, i32, i32) -> (i32, i32, i32)
      "scf.yield"(%6, %7, %8) : (i32, i32, i32) -> ()
    }) : (index, index, index, i32, i32, i32) -> (i32, i32, i32)
    "func.call"(%r_3) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_3, %r_4, %r_5) : (i32, i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()