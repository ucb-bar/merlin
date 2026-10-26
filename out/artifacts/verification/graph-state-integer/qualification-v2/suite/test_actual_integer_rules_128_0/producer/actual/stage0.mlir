"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i128, i128, i128) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i128) -> i128}> ({
  ^bb0(%x: i128):
    %zero = "arith.constant"() <{value = 0 : i128}> : () -> i128
    %one = "arith.constant"() <{value = 1 : i128}> : () -> i128
    %max = "arith.constant"() <{value = 170141183460469231731687303715884105727 : i128}> : () -> i128
    %two = "arith.constant"() <{value = 2 : i128}> : () -> i128
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i128, i128, i128) -> ()
    "func.return"(%n4) : (i128) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i128) -> i128}> ({
  ^bb0(%x: i128):
    "func.return"(%x) : (i128) -> ()
  }) : () -> ()
}) : () -> ()