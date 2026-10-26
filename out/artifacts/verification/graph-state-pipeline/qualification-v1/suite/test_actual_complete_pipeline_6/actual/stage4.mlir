"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i8, index) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i8, index) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i8) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%x: i8, %bias: i8):
    %lb = "arith.constant"() <{value = 2 : index}> : () -> index
    %ub = "arith.constant"() <{value = 34 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %out = "arith.constant"() <{value = 8 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 34 : index}> : () -> index
    %out_2 = "arith.constant"() <{value = 32 : index}> : () -> index
    %out_3 = "scf.for"(%lb, %out_1, %out_2, %x) ({
    ^bb0(%0: index, %1: i8):
      %2 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %out_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %out, %1) ({
      ^bb0(%5: index, %6: i8):
        %a = "arith.addi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %v = "func.call"(%a2, %5) <{callee = @partial}> : (i8, index) -> i8
        %7 = "arith.constant"() <{value = 2 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %v_1 = "func.call"(%a2_1, %8) <{callee = @partial}> : (i8, index) -> i8
        %9 = "arith.constant"() <{value = 4 : index}> : () -> index
        %10 = "arith.addi"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_2 = "arith.addi"(%v_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %v_2 = "func.call"(%a2_2, %10) <{callee = @partial}> : (i8, index) -> i8
        %11 = "arith.constant"() <{value = 6 : index}> : () -> index
        %12 = "arith.addi"(%5, %11) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_3 = "arith.addi"(%v_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %a2_3 = "arith.addi"(%a_3, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %v_3 = "func.call"(%a2_3, %12) <{callee = @partial}> : (i8, index) -> i8
        "scf.yield"(%v_3) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%4) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.call"(%out_3) <{callee = @final_observe}> : (i8) -> ()
    "func.return"(%out_3) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()