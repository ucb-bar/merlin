"builtin.module"() ({
  "func.func"() <{sym_name = "step", function_type = (index, i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "observe", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "altered", function_type = (i8) -> i8, sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lb = "arith.constant"() <{value = 7 : index}> : () -> index
    %ub = "arith.constant"() <{value = 16 : index}> : () -> index
    %st = "arith.constant"() <{value = 3 : index}> : () -> index
    %r = "arith.constant"() <{value = 7 : index}> : () -> index
    %sum = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %next = "func.call"(%r, %sum) <{callee = @step}> : (index, i8) -> i8
    %r_1 = "arith.constant"() <{value = 10 : index}> : () -> index
    %sum_1 = "arith.addi"(%next, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %next_1 = "func.call"(%r_1, %sum_1) <{callee = @step}> : (index, i8) -> i8
    %r_2 = "arith.constant"() <{value = 13 : index}> : () -> index
    %sum_2 = "arith.addi"(%next_1, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %next_2 = "func.call"(%r_2, %sum_2) <{callee = @step}> : (index, i8) -> i8
    %seen = "func.call"(%next_2) <{callee = @observe}> : (i8) -> i8
    "func.return"(%a) : (i8) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i8) -> i8}> ({
  ^bb0(%x: i8):
    "func.return"(%x) : (i8) -> ()
  }) : () -> ()
}) : () -> ()