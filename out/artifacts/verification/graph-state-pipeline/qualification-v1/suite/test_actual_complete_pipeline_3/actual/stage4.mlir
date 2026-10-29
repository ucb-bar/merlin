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
    %ub = "arith.constant"() <{value = 27 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %out = "arith.constant"() <{value = 4 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 28 : index}> : () -> index
    %out_2 = "arith.constant"() <{value = 8 : index}> : () -> index
    %out_3 = "scf.for"(%lb, %out_1, %out_2, %x) ({
    ^bb0(%0: index, %1: i16):
      %2 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %out_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %out, %1) ({
      ^bb0(%5: index, %6: i16):
        %a = "arith.addi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %v = "func.call"(%a2, %5) <{callee = @partial}> : (i16, index) -> i16
        %7 = "arith.constant"() <{value = 2 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %v_1 = "func.call"(%a2_1, %8) <{callee = @partial}> : (i16, index) -> i16
        "scf.yield"(%v_1) : (i16) -> ()
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