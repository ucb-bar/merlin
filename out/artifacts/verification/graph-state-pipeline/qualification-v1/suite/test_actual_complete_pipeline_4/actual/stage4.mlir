"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i16, index) -> i16, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i16, index) -> i16, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i16, i16) -> i16}> ({
  ^bb0(%x: i16, %bias: i16):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 12 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %out = "arith.constant"() <{value = 3 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 12 : index}> : () -> index
    %out_2 = "arith.constant"() <{value = 6 : index}> : () -> index
    %out_3 = "scf.for"(%lb, %out_1, %out_2, %x) ({
    ^bb0(%0: index, %1: i16):
      %2 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %out_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %out, %1) ({
      ^bb0(%5: index, %6: i16):
        %a = "arith.addi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %div = "arith.divsi"(%a2, %bias) : (i16, i16) -> i16
        %v = "func.call"(%div, %5) <{callee = @partial}> : (i16, index) -> i16
        %7 = "arith.constant"() <{value = 1 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %div_1 = "arith.divsi"(%a2_1, %bias) : (i16, i16) -> i16
        %v_1 = "func.call"(%div_1, %8) <{callee = @partial}> : (i16, index) -> i16
        %9 = "arith.constant"() <{value = 2 : index}> : () -> index
        %10 = "arith.addi"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_2 = "arith.addi"(%v_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %div_2 = "arith.divsi"(%a2_2, %bias) : (i16, i16) -> i16
        %v_2 = "func.call"(%div_2, %10) <{callee = @partial}> : (i16, index) -> i16
        "scf.yield"(%v_2) : (i16) -> ()
      }) : (index, index, index, i16) -> i16
      "scf.yield"(%4) : (i16) -> ()
    }) : (index, index, index, i16) -> i16
    "func.call"(%out_3) <{callee = @final_observe}> : (i16) -> ()
    "func.return"(%out_3) : (i16) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
}) : () -> ()