"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i8, i8, i8) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    %zero = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %one = "arith.constant"() <{value = 1 : i8}> : () -> i8
    %max = "arith.constant"() <{value = 127 : i8}> : () -> i8
    %two = "arith.constant"() <{value = 2 : i8}> : () -> i8
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i8, i8, i8) -> ()
    "func.return"(%n4) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()