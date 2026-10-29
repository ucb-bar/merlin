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
    %r = "scf.for"(%lb, %ub, %st, %x) ({
    ^bb0(%i: index, %c: i32):
      %a = "arith.addi"(%c, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %b = "func.call"(%a, %i) <{callee = @step}> : (i32, index) -> i32
      "scf.yield"(%b) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.call"(%r) <{callee = @watch}> : (i32) -> ()
    "func.return"(%r) : (i32) -> ()
  }) : () -> ()
}) : () -> ()