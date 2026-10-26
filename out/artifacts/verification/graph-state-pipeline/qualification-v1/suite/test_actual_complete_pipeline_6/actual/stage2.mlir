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
    %out = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%iv: index, %acc: i8):
      %a = "arith.addi"(%acc, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %v = "func.call"(%a2, %iv) <{callee = @partial}> : (i8, index) -> i8
      "scf.yield"(%v) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.call"(%out) <{callee = @final_observe}> : (i8) -> ()
    "func.return"(%out) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()