"builtin.module"() ({
  "func.func"() <{sym_name = "probe", function_type = (memref<2x3xi32>, memref<2x3xi32>, memref<1xi32>) -> (index, index, i32)}> ({
  ^bb0(%a: memref<2x3xi32>, %b: memref<2x3xi32>, %sink: memref<1xi32>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %one = "arith.constant"() <{value = 1 : index}> : () -> index
    %axis = "arith.constant"() <{value = 1 : index}> : () -> index
    %other = "arith.constant"() <{value = 0 : index}> : () -> index
    %d0 = "arith.constant"() <{value = 3 : index}> : () -> index
    %d1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %value = "memref.load"(%a, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    "memref.store"(%value, %sink, %zero) : (i32, memref<1xi32>, index) -> ()
    "func.return"(%d0, %d1, %value) : (index, index, i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%value: i32):
    "func.return"(%value) : (i32) -> ()
  }) : () -> ()
}) : () -> ()