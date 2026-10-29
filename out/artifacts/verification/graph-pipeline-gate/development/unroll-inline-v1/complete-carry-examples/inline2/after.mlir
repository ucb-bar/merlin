"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (index, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i8, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "altered", function_type = (i8, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 2 : index}> : () -> index
    %ub = "arith.constant"() <{value = 8 : index}> : () -> index
    %st = "arith.constant"() <{value = 3 : index}> : () -> index
    %r = "arith.constant"() <{value = 2 : index}> : () -> index
    %sum = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %next = "func.call"(%r, %sum) <{callee = @step}> : (index, i8) -> i8
    %r_1 = "arith.constant"() <{value = 5 : index}> : () -> index
    %sum_1 = "arith.addi"(%next, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %next_1 = "func.call"(%r_1, %sum_1) <{callee = @step}> : (index, i8) -> i8
    %seen = "func.call"(%next_1, %b) <{callee = @observe}> : (i8, i8) -> i8
    "func.return"(%seen) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()