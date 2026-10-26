"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i1, i1, i1) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i1) -> i1}> ({
  ^bb0(%x: i1):
    %zero = "arith.constant"() <{value = false}> : () -> i1
    %one = "arith.constant"() <{value = true}> : () -> i1
    %max = "arith.constant"() <{value = false}> : () -> i1
    %two = "arith.constant"() <{value = true}> : () -> i1
    %fold = "arith.constant"() <{value = true}> : () -> i1
    %chain = "arith.constant"() <{value = false}> : () -> i1
    %product = "arith.constant"() <{value = false}> : () -> i1
    %n3 = "arith.muli"(%one, %x) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i1, i1, i1) -> ()
    "func.return"(%n4) : (i1) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i1) -> i1}> ({
  ^bb0(%x: i1):
    "func.return"(%x) : (i1) -> ()
  }) : () -> ()
}) : () -> ()