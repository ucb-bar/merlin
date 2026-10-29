"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i256, i256, i256) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i256) -> i256}> ({
  ^bb0(%x: i256):
    %zero = "arith.constant"() <{value = 0 : i256}> : () -> i256
    %one = "arith.constant"() <{value = 1 : i256}> : () -> i256
    %max = "arith.constant"() <{value = 57896044618658097711785492504343953926634992332820282019728792003956564819967 : i256}> : () -> i256
    %two = "arith.constant"() <{value = 2 : i256}> : () -> i256
    %fold = "arith.constant"() <{value = -57896044618658097711785492504343953926634992332820282019728792003956564819967 : i256}> : () -> i256
    %chain = "arith.constant"() <{value = -57896044618658097711785492504343953926634992332820282019728792003956564819968 : i256}> : () -> i256
    %product = "arith.constant"() <{value = 0 : i256}> : () -> i256
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i256, i256, i256) -> ()
    "func.return"(%x) : (i256) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i256) -> i256}> ({
  ^bb0(%x: i256):
    "func.return"(%x) : (i256) -> ()
  }) : () -> ()
}) : () -> ()