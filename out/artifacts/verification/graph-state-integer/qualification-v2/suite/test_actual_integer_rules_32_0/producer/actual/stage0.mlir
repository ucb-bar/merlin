"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i32, i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %max = "arith.constant"() <{value = 2147483647 : i32}> : () -> i32
    %two = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i32, i32, i32) -> ()
    "func.return"(%n4) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()