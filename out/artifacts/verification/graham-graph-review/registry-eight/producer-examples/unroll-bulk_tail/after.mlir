"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, i32, memref<?xi32>) -> (i32, i32)}> ({
  ^bb0(%input0: i32, %input1: i32, %capture: i32, %memory: memref<?xi32>):
    %lower = "arith.constant"() <{value = 3 : index}> : () -> index
    %upper = "arith.constant"() <{value = 25 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %result = "arith.constant"() <{value = 6 : index}> : () -> index
    %result_1 = "arith.constant"() <{value = 21 : index}> : () -> index
    %result_2, %result_3 = "scf.for"(%lower, %result_1, %result, %input0, %input1) ({
    ^bb1(%0: index, %1: i32, %2: i32):
      %next0 = "arith.addi"(%1, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %next1 = "arith.addi"(%2, %next0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%next0, %memory, %0) : (i32, memref<?xi32>, index) -> ()
      %3 = "arith.constant"() <{value = 2 : index}> : () -> index
      %4 = "arith.addi"(%0, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %next0_1 = "arith.addi"(%next0, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %next1_1 = "arith.addi"(%next1, %next0_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%next0_1, %memory, %4) : (i32, memref<?xi32>, index) -> ()
      %5 = "arith.constant"() <{value = 4 : index}> : () -> index
      %6 = "arith.addi"(%0, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %next0_2 = "arith.addi"(%next0_1, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %next1_2 = "arith.addi"(%next1_1, %next0_2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%next0_2, %memory, %6) : (i32, memref<?xi32>, index) -> ()
      "scf.yield"(%next0_2, %next1_2) : (i32, i32) -> ()
    }) : (index, index, index, i32, i32) -> (i32, i32)
    %result_4, %result_5 = "scf.for"(%result_1, %upper, %step, %result_2, %result_3) ({
    ^bb2(%7: index, %8: i32, %9: i32):
      %next0_3 = "arith.addi"(%8, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %next1_3 = "arith.addi"(%9, %next0_3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%next0_3, %memory, %7) : (i32, memref<?xi32>, index) -> ()
      "scf.yield"(%next0_3, %next1_3) : (i32, i32) -> ()
    }) : (index, index, index, i32, i32) -> (i32, i32)
    %observer = "arith.addi"(%result_4, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%observer, %result_5) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%value: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%value, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
}) : () -> ()