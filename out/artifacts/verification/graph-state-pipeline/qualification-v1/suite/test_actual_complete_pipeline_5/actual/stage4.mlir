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
    %out_2 = "arith.constant"() <{value = 18 : index}> : () -> index
    %out_3, %out_4, %out_5 = "scf.for"(%lb, %out_1, %out_2, %x, %bias, %x) ({
    ^bb0(%0: index, %1: i64, %2: i64, %3: i64):
      %4 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %5 = "arith.minui"(%4, %out_1) : (index, index) -> index
      %6, %7, %8 = "scf.for"(%0, %5, %out, %1, %2, %3) ({
      ^bb0(%9: index, %10: i64, %11: i64, %12: i64):
        %a = "arith.addi"(%invariant, %10) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2, %9) <{callee = @watch}> : (i64, index) -> ()
        %v, %v_1, %v_2 = "func.call"(%a2, %11, %12, %9) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
        %next = "arith.subi"(%v_1, %10) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %last = "arith.addi"(%v_2, %11) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %13 = "arith.constant"() <{value = 2 : index}> : () -> index
        %14 = "arith.addi"(%9, %13) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_1 = "arith.addi"(%invariant, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2_1, %14) <{callee = @watch}> : (i64, index) -> ()
        %v_3, %v_4, %v_5 = "func.call"(%a2_1, %next, %last, %14) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
        %next_1 = "arith.subi"(%v_4, %v) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %last_1 = "arith.addi"(%v_5, %next) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %15 = "arith.constant"() <{value = 4 : index}> : () -> index
        %16 = "arith.addi"(%9, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %a_2 = "arith.addi"(%invariant, %v_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %a2_2 = "arith.addi"(%a_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%a2_2, %16) <{callee = @watch}> : (i64, index) -> ()
        %v_6, %v_7, %v_8 = "func.call"(%a2_2, %next_1, %last_1, %16) <{callee = @partial}> : (i64, i64, i64, index) -> (i64, i64, i64)
        %next_2 = "arith.subi"(%v_7, %v_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %last_2 = "arith.addi"(%v_8, %next_1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "scf.yield"(%v_6, %next_2, %last_2) : (i64, i64, i64) -> ()
      }) : (index, index, index, i64, i64, i64) -> (i64, i64, i64)
      "scf.yield"(%6, %7, %8) : (i64, i64, i64) -> ()
    }) : (index, index, index, i64, i64, i64) -> (i64, i64, i64)
    "func.call"(%out_3, %out_4, %out_5) <{callee = @final_observe}> : (i64, i64, i64) -> ()
    "func.return"(%out_5, %out_3, %out_4) : (i64, i64, i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()