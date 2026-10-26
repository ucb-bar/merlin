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
    %out_2 = "arith.constant"() <{value = 18 : index}> : () -> index
    %out_3, %out_4 = "scf.for"(%lb, %out_1, %out_2, %x, %bias) ({
    ^bb0(%0: index, %1: i64, %2: i64):
      %3 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %4 = "arith.minui"(%3, %out_1) : (index, index) -> index
      %5, %6 = "scf.for"(%0, %4, %out, %1, %2) ({
      ^bb0(%7: index, %8: i64, %9: i64):
        %a = "arith.addi"(%invariant, %8) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2, %7) <{callee = @watch}> : (i64, index) -> ()
        %v, %v_1 = "func.call"(%a2, %9, %7) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
        %next = "arith.subi"(%v_1, %8) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %10 = "arith.constant"() <{value = 2 : index}> : () -> index
        %11 = "arith.addi"(%7, %10) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_1 = "arith.addi"(%invariant, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2_1, %11) <{callee = @watch}> : (i64, index) -> ()
        %v_2, %v_3 = "func.call"(%a2_1, %next, %11) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
        %next_1 = "arith.subi"(%v_3, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %12 = "arith.constant"() <{value = 4 : index}> : () -> index
        %13 = "arith.addi"(%7, %12) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_2 = "arith.addi"(%invariant, %v_2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2_2, %13) <{callee = @watch}> : (i64, index) -> ()
        %v_4, %v_5 = "func.call"(%a2_2, %next_1, %13) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
        %next_2 = "arith.subi"(%v_5, %v_2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "scf.yield"(%v_4, %next_2) : (i64, i64) -> ()
      }) : (index, index, index, i64, i64) -> (i64, i64)
      "scf.yield"(%5, %6) : (i64, i64) -> ()
    }) : (index, index, index, i64, i64) -> (i64, i64)
    "func.call"(%out_3, %out_4) <{callee = @final_observe}> : (i64, i64) -> ()
    "func.return"(%out_4, %out_3) : (i64, i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()