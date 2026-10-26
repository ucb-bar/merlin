"builtin.module"() ({
  "func.func"() <{sym_name = "partial", function_type = (i16, index) -> i16, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i16, index) -> i16, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final_observe", function_type = (i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i16, i16) -> i16}> ({
  ^bb0(%x: i16, %bias: i16):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 12 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %dead = "arith.addi"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %out = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%iv: index, %acc: i16):
      %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %duplicate = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %a = "arith.addi"(%acc, %duplicate) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %div = "arith.divsi"(%a2, %bias) : (i16, i16) -> i16
      %v = "func.call"(%div, %iv) <{callee = @partial}> : (i16, index) -> i16
      "scf.yield"(%v) : (i16) -> ()
    }) : (index, index, index, i16) -> i16
    "func.call"(%out) <{callee = @final_observe}> : (i16) -> ()
    "func.return"(%out) : (i16) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
}) : () -> ()