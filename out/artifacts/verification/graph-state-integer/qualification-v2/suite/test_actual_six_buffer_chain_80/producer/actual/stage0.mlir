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
    %zero = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %one = "arith.constant"() <{value = 1 : i8}> : () -> i8
    %a = "arith.constant"() <{value = -3 : i8}> : () -> i8
    %b = "arith.constant"() <{value = 3 : i8}> : () -> i8
    %foldedzero = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %later = "arith.addi"(%x, %foldedzero) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %dead = "arith.subi"(%x, %bias) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %out = "scf.for"(%lb, %ub, %st, %later) ({
    ^bb0(%iv: index, %acc: i8):
      %invariant = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %duplicate = "arith.addi"(%bias, %bias) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %neutral = "arith.muli"(%acc, %one) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %sum = "arith.addi"(%neutral, %duplicate) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %sum2 = "arith.addi"(%sum, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      "func.call"(%sum2, %iv) <{callee = @watch}> : (i8, index) -> ()
      %v = "func.call"(%sum2, %iv) <{callee = @provider}> : (i8, index) -> i8
      %next = "arith.subi"(%v, %zero) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      "scf.yield"(%next) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.call"(%out) <{callee = @final}> : (i8) -> ()
    "func.return"(%out) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()