"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, i32, i32, memref<?xi32>) -> (i32, i32, i32)}> ({
  ^bb0(%input0: i32, %input1: i32, %input2: i32, %capture: i32, %memory: memref<?xi32>):
    %lower = "arith.constant"() <{value = 1 : index}> : () -> index
    %upper = "arith.constant"() <{value = 8 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %result = "arith.constant"() <{value = 1 : index}> : () -> index
    %next0 = "arith.addi"(%input0, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next1 = "arith.addi"(%input1, %next0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next2 = "arith.addi"(%input2, %next1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%next0, %memory, %result) : (i32, memref<?xi32>, index) -> ()
    %result_1 = "arith.constant"() <{value = 3 : index}> : () -> index
    %next0_1 = "arith.addi"(%next0, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next1_1 = "arith.addi"(%next1, %next0_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next2_1 = "arith.addi"(%next2, %next1_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%next0_1, %memory, %result_1) : (i32, memref<?xi32>, index) -> ()
    %result_2 = "arith.constant"() <{value = 5 : index}> : () -> index
    %next0_2 = "arith.addi"(%next0_1, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next1_2 = "arith.addi"(%next1_1, %next0_2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next2_2 = "arith.addi"(%next2_1, %next1_2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%next0_2, %memory, %result_2) : (i32, memref<?xi32>, index) -> ()
    %result_3 = "arith.constant"() <{value = 7 : index}> : () -> index
    %next0_3 = "arith.addi"(%next0_2, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next1_3 = "arith.addi"(%next1_2, %next0_3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %next2_3 = "arith.addi"(%next2_2, %next1_3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%next0_3, %memory, %result_3) : (i32, memref<?xi32>, index) -> ()
    %observer = "arith.addi"(%next0_3, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%observer, %next1_3, %next2_3) : (i32, i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%value: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%value, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
}) : () -> ()