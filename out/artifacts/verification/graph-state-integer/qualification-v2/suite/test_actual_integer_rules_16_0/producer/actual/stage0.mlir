"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i16, i16, i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    %zero = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %one = "arith.constant"() <{value = 1 : i16}> : () -> i16
    %max = "arith.constant"() <{value = 32767 : i16}> : () -> i16
    %two = "arith.constant"() <{value = 2 : i16}> : () -> i16
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i16, i16, i16) -> ()
    "func.return"(%n4) : (i16) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
}) : () -> ()