"builtin.module"() ({
  "func.func"() <{sym_name = "provider", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32, index) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "final", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> i32}> ({
  ^bb0(%x: i32, %bias: i32):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 12 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %a = "arith.constant"() <{value = -3 : i32}> : () -> i32
    %b = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %foldedzero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %dead = "arith.subi"(%x, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %out = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%iv: index, %acc: i32):
      %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %duplicate = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %duplicate) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "func.call"(%sum2, %iv) <{callee = @watch}> : (i32, index) -> ()
      %v = "func.call"(%sum2, %iv) <{callee = @provider}> : (i32, index) -> i32
      "scf.yield"(%v) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%out) <{callee = @final}> : (i32) -> ()
    "func.return"(%out) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()