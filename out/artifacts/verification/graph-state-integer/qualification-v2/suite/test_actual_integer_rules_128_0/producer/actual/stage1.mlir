"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i128, i128, i128) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i128) -> i128}> ({
  ^bb0(%x: i128):
    %zero = "arith.constant"() <{value = 0 : i128}> : () -> i128
    %one = "arith.constant"() <{value = 1 : i128}> : () -> i128
    %max = "arith.constant"() <{value = 170141183460469231731687303715884105727 : i128}> : () -> i128
    %two = "arith.constant"() <{value = 2 : i128}> : () -> i128
    %fold = "arith.constant"() <{value = -170141183460469231731687303715884105727 : i128}> : () -> i128
    %chain = "arith.constant"() <{value = -170141183460469231731687303715884105728 : i128}> : () -> i128
    %product = "arith.constant"() <{value = 0 : i128}> : () -> i128
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i128, i128, i128) -> ()
    "func.return"(%x) : (i128) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i128) -> i128}> ({
  ^bb0(%x: i128):
    "func.return"(%x) : (i128) -> ()
  }) : () -> ()
}) : () -> ()