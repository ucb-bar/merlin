"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i64, i64, i64, index) -> (i64, i64, i64), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i64, i64, i64, index) -> (i64, i64, i64), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i64, index) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i64, i64, i64) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i64, i64) -> (i64, i64, i64)}> ({
  ^bb0(%x: i64, %bias: i64):
    %lb = "arith.constant"() <{value = 3 : index}> : () -> index
    %ub = "arith.constant"() <{value = 27 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %invariant = "arith.muli"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %out = "arith.constant"() <{value = 6 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 27 : index}> : () -> index
    %out_2, %out_3, %out_4 = "scf.for"(%lb, %out_1, %out, %x, %bias, %x) ({
    ^bb0(%0: index, %1: i64, %2: i64, %3: i64):
      %a = "arith.addi"(%invariant, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2, %0) <{callee = @watch}> : (i64, index) -> ()
      %v, %v_1, %v_2 = "func.call"(%a2, %2, %3, %0) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
      %next = "arith.subi"(%v_1, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %last = "arith.addi"(%v_2, %2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %4 = "arith.constant"() <{value = 2 : index}> : () -> index
      %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_1 = "arith.addi"(%invariant, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2_1, %5) <{callee = @watch}> : (i64, index) -> ()
      %v_3, %v_4, %v_5 = "func.call"(%a2_1, %next, %last, %5) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
      %next_1 = "arith.subi"(%v_4, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %last_1 = "arith.addi"(%v_5, %next) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %6 = "arith.constant"() <{value = 4 : index}> : () -> index
      %7 = "arith.addi"(%0, %6) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_2 = "arith.addi"(%invariant, %v_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2_2, %7) <{callee = @watch}> : (i64, index) -> ()
      %v_6, %v_7, %v_8 = "func.call"(%a2_2, %next_1, %last_1, %7) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
      %next_2 = "arith.subi"(%v_7, %v_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %last_2 = "arith.addi"(%v_8, %next_1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "scf.yield"(%v_6, %next_2, %last_2) : (i64, i64, i64) -> ()
    }) : (index, index, index, i64, i64, i64) -> (i64, i64, i64)
    "func.call"(%out_2, %out_3, %out_4) <{callee = @final_observe}> : (i64, i64, i64) -> ()
    "func.return"(%out_4, %out_2, %out_3) : (i64, i64, i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()