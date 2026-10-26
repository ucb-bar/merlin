"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i16, i16, i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    %zero = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %one = "arith.constant"() <{value = 1 : i16}> : () -> i16
    %max = "arith.constant"() <{value = 32767 : i16}> : () -> i16
    %two = "arith.constant"() <{value = 2 : i16}> : () -> i16
    %fold = "arith.constant"() <{value = -32767 : i16}> : () -> i16
    %chain = "arith.constant"() <{value = -32768 : i16}> : () -> i16
    %product = "arith.constant"() <{value = 0 : i16}> : () -> i16
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i16, i16, i16) -> ()
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
}) : () -> ()