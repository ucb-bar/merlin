"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i64, i64, i64) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    %zero = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %one = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %max = "arith.constant"() <{value = 9223372036854775807 : i64}> : () -> i64
    %two = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %fold = "arith.constant"() <{value = -9223372036854775807 : i64}> : () -> i64
    %chain = "arith.constant"() <{value = -9223372036854775808 : i64}> : () -> i64
    %product = "arith.constant"() <{value = 0 : i64}> : () -> i64
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i64, i64, i64) -> ()
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i64) -> i64}> ({
  ^bb0(%x: i64):
    "func.return"(%x) : (i64) -> ()
  }) : () -> ()
}) : () -> ()