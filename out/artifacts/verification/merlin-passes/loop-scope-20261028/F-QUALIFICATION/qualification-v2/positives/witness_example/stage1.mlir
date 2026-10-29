"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (i32, index) -> i32, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i32) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i32, i32) -> i32}> ({
  ^bb0(%x: i32, %y: i32):
    %lb = "arith.constant"() <{value = 3 : index}> : () -> index
    %ub = "arith.constant"() <{value = 27 : index}> : () -> index
    %st = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 4 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 27 : index}> : () -> index
    %r_2 = "scf.for"(%lb, %r_1, %r, %x) ({
    ^bb0(%0: index, %1: i32):
      %a = "arith.addi"(%1, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b = "func.call"(%a, %0) <{callee = @step}> : (i32, index) -> i32
      %2 = "arith.constant"() <{value = 2 : index}> : () -> index
      %3 = "arith.addi"(%0, %2) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %a_1 = "arith.addi"(%b, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b_1 = "func.call"(%a_1, %3) <{callee = @step}> : (i32, index) -> i32
      "scf.yield"(%b_1) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%r_2) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r_2) : (i32) -> ()
  }) : () -> ()
}) : () -> ()