"builtin.module"() ({
  "func.func"() <{sym_name = "provider", function_type = (i8, index) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i8, index) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final", function_type = (i8) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%x: i8, %bias: i8):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 12 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %out = "arith.constant"() <{value = 2 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 12 : index}> : () -> index
    %out_2 = "arith.constant"() <{value = 6 : index}> : () -> index
    %out_3 = "scf.for"(%lb, %out_1, %out_2, %x) ({
    ^bb0(%0: index, %1: i8):
      %2 = "arith.addi"(%0, %out_2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %out_1) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %out, %1) ({
      ^bb0(%5: index, %6: i8):
        %sum = "arith.addi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "func.call"(%sum2, %5) <{callee = @watch}> : (i8, index) -> ()
        %v = "func.call"(%sum2, %5) <{callee = @provider}> : (i8, index) -> i8
        %7 = "arith.constant"() <{value = 1 : index}> : () -> index
        %8 = "arith.addi"(%5, %7) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %sum_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %sum2_1 = "arith.addi"(%sum_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "func.call"(%sum2_1, %8) <{callee = @watch}> : (i8, index) -> ()
        %v_1 = "func.call"(%sum2_1, %8) <{callee = @provider}> : (i8, index) -> i8
        "scf.yield"(%v_1) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%4) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.call"(%out_3) <{callee = @final}> : (i8) -> ()
    "func.return"(%out_3) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()