"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, i32, memref<?xi32>) -> (i32, i32)}> ({
  ^bb0(%input0: i32, %input1: i32, %capture: i32, %memory: memref<?xi32>):
    %lower = "arith.constant"() <{value = 7 : index}> : () -> index
    %upper = "arith.constant"() <{value = 7 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %observer = "arith.addi"(%input0, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%observer, %input1) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%value: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%value, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
}) : () -> ()