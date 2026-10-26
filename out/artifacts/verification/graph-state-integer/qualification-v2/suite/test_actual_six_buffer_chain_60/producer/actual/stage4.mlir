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
    %out_2 = "scf.for"(%lb, %out_1, %out, %x) ({
    ^bb0(%0: index, %1: i64):
      %sum = "arith.addi"(%1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%sum2, %0) <{callee = @watch}> : (i64, index) -> ()
      %v = "func.call"(%sum2, %0) <{callee = @provider}> : (i64, index) -> i64
      %2 = "arith.constant"() <{value = 1 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %sum_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum2_1 = "arith.addi"(%sum_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%sum2_1, %3) <{callee = @watch}> : (i64, index) -> ()
      %v_1 = "func.call"(%sum2_1, %3) <{callee = @provider}> : (i64, index) -> i64
      %4 = "arith.constant"() <{value = 2 : index}> : () -> index
      %5 = "arith.addi"(%0, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %sum_2 = "arith.addi"(%v_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum2_2 = "arith.addi"(%sum_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%sum2_2, %5) <{callee = @watch}> : (i64, index) -> ()
      %v_2 = "func.call"(%sum2_2, %5) <{callee = @provider}> : (i64, index) -> i64
      %6 = "arith.constant"() <{value = 3 : index}> : () -> index
      %7 = "arith.addi"(%0, %6) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %sum_3 = "arith.addi"(%v_2, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum2_3 = "arith.addi"(%sum_3, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%sum2_3, %7) <{callee = @watch}> : (i64, index) -> ()
      %v_3 = "func.call"(%sum2_3, %7) <{callee = @provider}> : (i64, index) -> i64
      "scf.yield"(%v_3) : (i64) -> ()
    }) : (index, index, index, i64) -> i64
    "func.call"(%out_2) <{callee = @final}> : (i64) -> ()
    "func.return"(%out_2) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()