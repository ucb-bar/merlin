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
    %zero = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %one = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %a = "arith.constant"() <{value = -3 : i64}> : () -> i64
    %b = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %foldedzero = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %dead = "arith.subi"(%x, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %out = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%iv: index, %acc: i64):
      %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %duplicate = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum = "arith.addi"(%acc, %duplicate) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "func.call"(%sum2, %iv) <{callee = @watch}> : (i64, index) -> ()
      %v = "func.call"(%sum2, %iv) <{callee = @provider}> : (i64, index) -> i64
      "scf.yield"(%v) : (i64) -> ()
    }) : (index, index, index, i64) -> i64
    "func.call"(%out) <{callee = @final}> : (i64) -> ()
    "func.return"(%out) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()