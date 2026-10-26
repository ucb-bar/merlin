"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i64, i64, index) -> (i64, i64), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i64, i64, index) -> (i64, i64), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i64, index) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i64, i64) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i64, i64) -> (i64, i64)}> ({
  ^bb0(%x: i64, %bias: i64):
    %lb = "arith.constant"() <{value = 3 : index}> : () -> index
    %ub = "arith.constant"() <{value = 27 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %invariant = "arith.muli"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %out = "arith.constant"() <{value = 6 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 27 : index}> : () -> index
    %out_2, %out_3 = "scf.for"(%lb, %out_1, %out, %x, %bias) ({
    ^bb0(%0: index, %1: i64, %2: i64):
      %a = "arith.addi"(%invariant, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2, %0) <{callee = @watch}> : (i64, index) -> ()
      %v, %v_1 = "func.call"(%a2, %2, %0) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
      %next = "arith.subi"(%v_1, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %3 = "arith.constant"() <{value = 2 : index}> : () -> index
      %4 = "arith.addi"(%0, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_1 = "arith.addi"(%invariant, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2_1, %4) <{callee = @watch}> : (i64, index) -> ()
      %v_2, %v_3 = "func.call"(%a2_1, %next, %4) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
      %next_1 = "arith.subi"(%v_3, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %5 = "arith.constant"() <{value = 4 : index}> : () -> index
      %6 = "arith.addi"(%0, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_2 = "arith.addi"(%invariant, %v_2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2_2, %6) <{callee = @watch}> : (i64, index) -> ()
      %v_4, %v_5 = "func.call"(%a2_2, %next_1, %6) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
      %next_2 = "arith.subi"(%v_5, %v_2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "scf.yield"(%v_4, %next_2) : (i64, i64) -> ()
    }) : (index, index, index, i64, i64) -> (i64, i64)
    "func.call"(%out_2, %out_3) <{callee = @final_observe}> : (i64, i64) -> ()
    "func.return"(%out_3, %out_2) : (i64, i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()