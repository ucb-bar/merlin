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
    %out = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%iv: index, %acc: i32):
      %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %a = "arith.addi"(%acc, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %v = "func.call"(%a2, %iv) <{callee = @partial}> : (i32, index) -> i32
      "scf.yield"(%v) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%out) <{callee = @final_observe}> : (i32) -> ()
    "func.return"(%out) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()