"builtin.module"() ({
  "func.func"() <{sym_name = "watch", function_type = (i1, i1, i1) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i1) -> i1}> ({
  ^bb0(%x: i1):
    %zero = "arith.constant"() <{value = false}> : () -> i1
    %one = "arith.constant"() <{value = true}> : () -> i1
    %max = "arith.constant"() <{value = false}> : () -> i1
    %two = "arith.constant"() <{value = true}> : () -> i1
    %fold = "arith.addi"(%max, %two) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %chain = "arith.subi"(%fold, %one) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %product = "arith.muli"(%chain, %two) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n0 = "arith.addi"(%zero, %x) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n1 = "arith.addi"(%n0, %zero) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n2 = "arith.subi"(%n1, %zero) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n3 = "arith.muli"(%one, %n2) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    %n4 = "arith.muli"(%n3, %one) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
    "func.call"(%fold, %chain, %product) <{callee = @watch}> : (i1, i1, i1) -> ()
    "func.return"(%n4) : (i1) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i1) -> i1}> ({
  ^bb0(%x: i1):
    "func.return"(%x) : (i1) -> ()
  }) : () -> ()
}) : () -> ()