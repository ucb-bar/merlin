"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i256, i256, i256) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i256) -> i256}> ({
  ^bb0(%x: i256):
    %zero = "arith.constant"() <{value = 0 : i256}> : () -> i256
    %one = "arith.constant"() <{value = 1 : i256}> : () -> i256
    %max = "arith.constant"() <{value = 57896044618658097711785492504343953926634992332820282019728792003956564819967 : i256}> : () -> i256
    %two = "arith.constant"() <{value = 2 : i256}> : () -> i256
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i256, i256) -> i256
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i256, i256, i256) -> ()
    "func.return"(%n4) : (i256) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i256) -> i256}> ({
  ^bb0(%x: i256):
    "func.return"(%x) : (i256) -> ()
  }) : () -> ()
}) : () -> ()