"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> i32}> ({
  ^bb0(%x: i32, %bias: i32):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 24 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %out = "arith.constant"() <{value = 2 : index}> : () -> index
    %out_1 = "arith.constant"() <{value = 24 : index}> : () -> index
    %out_2 = "scf.for"(%lb, %out_1, %out, %x) ({
    ^bb0(%0: index, %1: i32):
      %a = "arith.addi"(%1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v = "func.call"(%a2, %0) <{callee = @partial}> : (i32, index) -> i32
      %2 = "arith.constant"() <{value = 1 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_1 = "arith.addi"(%v, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %a2_1 = "arith.addi"(%a_1, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v_1 = "func.call"(%a2_1, %3) <{callee = @partial}> : (i32, index) -> i32
      "scf.yield"(%v_1) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%out_2) <{callee = @final_observe}> : (i32) -> ()
    "func.return"(%out_2) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()