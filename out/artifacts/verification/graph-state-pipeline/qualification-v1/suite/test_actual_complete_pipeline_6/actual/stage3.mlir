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
    %out_2 = "scf.for"(%lb, %out_1, %out, %x) ({
    ^bb0(%0: index, %1: i8):
      %a = "arith.addi"(%1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %v = "func.call"(%a2, %0) <{callee = @partial}> : (i8, index) -> i8
      %2 = "arith.constant"() <{value = 2 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %v_1 = "func.call"(%a2_1, %3) <{callee = @partial}> : (i8, index) -> i8
      %4 = "arith.constant"() <{value = 4 : index}> : () -> index
      %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_2 = "arith.addi"(%v_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %v_2 = "func.call"(%a2_2, %5) <{callee = @partial}> : (i8, index) -> i8
      %6 = "arith.constant"() <{value = 6 : index}> : () -> index
      %7 = "arith.addi"(%0, %6) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_3 = "arith.addi"(%v_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %a2_3 = "arith.addi"(%a_3, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %v_3 = "func.call"(%a2_3, %7) <{callee = @partial}> : (i8, index) -> i8
      "scf.yield"(%v_3) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.call"(%out_2) <{callee = @final_observe}> : (i8) -> ()
    "func.return"(%out_2) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()