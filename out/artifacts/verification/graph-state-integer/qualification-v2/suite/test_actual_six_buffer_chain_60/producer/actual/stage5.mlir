"builtin.module"() ({
  "func.func"() <{sym_name = "provider", function_type = (i64, index) -> i64, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i64, index) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final", function_type = (i64) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i64, i64) -> i64}> ({
  ^bb0(%x: i64, %bias: i64):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 12 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %out = "arith.constant"() <{value = 4 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 12 : index}> : () -> index
    %out_2 = "arith.constant"() <{value = 8 : index}> : () -> index
    %out_3 = "scf.for"(%lb, %out_1, %out_2, %x) ({
    ^bb0(%0: index, %1: i64):
      %2 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %out_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %out, %1) ({
      ^bb0(%5: index, %6: i64):
        %sum = "arith.addi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%sum2, %5) <{callee = @watch}> : (i64, index) -> ()
        %v = "func.call"(%sum2, %5) <{callee = @provider}> : (i64, index) -> i64
        %7 = "arith.constant"() <{value = 1 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %sum_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %sum2_1 = "arith.addi"(%sum_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%sum2_1, %8) <{callee = @watch}> : (i64, index) -> ()
        %v_1 = "func.call"(%sum2_1, %8) <{callee = @provider}> : (i64, index) -> i64
        %9 = "arith.constant"() <{value = 2 : index}> : () -> index
        %10 = "arith.addi"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %sum_2 = "arith.addi"(%v_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %sum2_2 = "arith.addi"(%sum_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%sum2_2, %10) <{callee = @watch}> : (i64, index) -> ()
        %v_2 = "func.call"(%sum2_2, %10) <{callee = @provider}> : (i64, index) -> i64
        %11 = "arith.constant"() <{value = 3 : index}> : () -> index
        %12 = "arith.addi"(%5, %11) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %sum_3 = "arith.addi"(%v_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        %sum2_3 = "arith.addi"(%sum_3, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
        "func.call"(%sum2_3, %12) <{callee = @watch}> : (i64, index) -> ()
        %v_3 = "func.call"(%sum2_3, %12) <{callee = @provider}> : (i64, index) -> i64
        "scf.yield"(%v_3) : (i64) -> ()
      }) : (index, index, index, i64) -> i64
      "scf.yield"(%4) : (i64) -> ()
    }) : (index, index, index, i64) -> i64
    "func.call"(%out_3) <{callee = @final}> : (i64) -> ()
    "func.return"(%out_3) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()