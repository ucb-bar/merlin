"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, i32, memref<?xi32>) -> (i32, i32)}> ({
  ^bb0(%input0: i32, %input1: i32, %capture: i32, %memory: memref<?xi32>):
    %lower = "arith.constant"() <{value = 3 : index}> : () -> index
    %upper = "arith.constant"() <{value = 25 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %result, %result_1 = "scf.for"(%lower, %upper, %step, %input0, %input1) ({
    ^bb1(%iv: index, %carry0: i32, %carry1: i32):
      %next0 = "arith.addi"(%carry0, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %next1 = "arith.addi"(%carry1, %next0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%next0, %memory, %iv) : (i32, memref<?xi32>, index) -> ()
      "scf.yield"(%next0, %next1) : (i32, i32) -> ()
    }) : (index, index, index, i32, i32) -> (i32, i32)
    %observer = "arith.addi"(%result, %capture) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%observer, %result_1) : (i32, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%value: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%value, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
}) : () -> ()