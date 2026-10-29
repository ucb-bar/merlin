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
    %out, %out_1 = "scf.for"(%lb, %ub, %st, %x, %bias) ({
    ^bb0(%iv: index, %acc: i64, %other: i64):
      %a = "arith.addi"(%invariant, %acc) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %a2 = "arith.addi"(%a, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%a2, %iv) <{callee = @watch}> : (i64, index) -> ()
      %v, %v_1 = "func.call"(%a2, %other, %iv) <{callee = @partial}> : (i64, i64, index) -> (i64, i64)
      %next = "arith.subi"(%v_1, %acc) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "scf.yield"(%v, %next) : (i64, i64) -> ()
    }) : (index, index, index, i64, i64) -> (i64, i64)
    "func.call"(%out, %out_1) <{callee = @final_observe}> : (i64, i64) -> ()
    "func.return"(%out_1, %out) : (i64, i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()