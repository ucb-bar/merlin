"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i32, i32, i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %max = "arith.constant"() <{value = 2147483647 : i32}> : () -> i32
    %two = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %fold = "arith.constant"() <{value = -2147483647 : i32}> : () -> i32
    %chain = "arith.constant"() <{value = -2147483648 : i32}> : () -> i32
    %product = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i32, i32, i32) -> ()
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%x: i32):
    "func.return"(%x) : (i32) -> ()
  }) : () -> ()
}) : () -> ()