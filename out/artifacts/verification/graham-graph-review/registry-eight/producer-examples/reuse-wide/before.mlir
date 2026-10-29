"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i128, i128, memref<?xi128>) -> (i128, index)}> ({
  ^bb0(%seed0: i128, %seed1: i128, %caller: memref<?xi128>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %one = "arith.constant"() <{value = 1 : index}> : () -> index
    %bound0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %buffer0 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2xi128>
    "scf.for"(%zero, %bound0, %one) ({
    ^bb1(%iv0: index):
      "memref.store"(%seed0, %buffer0, %iv0) : (i128, memref<2xi128>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first0 = "memref.load"(%buffer0, %zero) : (memref<2xi128>, index) -> i128
    %updated0 = "arith.addi"(%first0, %seed0) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    "memref.store"(%updated0, %buffer0, %zero) : (i128, memref<2xi128>, index) -> ()
    %value0 = "memref.load"(%buffer0, %zero) : (memref<2xi128>, index) -> i128
    %dim0 = "memref.dim"(%buffer0, %zero) : (memref<2xi128>, index) -> index
    %observed0 = "func.call"(%value0, %caller) <{callee = @touch}> : (i128, memref<?xi128>) -> i128
    "memref.dealloc"(%buffer0) : (memref<2xi128>) -> ()
    %buffer1 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2xi128>
    "scf.for"(%zero, %bound0, %one) ({
    ^bb2(%iv1: index):
      "memref.store"(%seed1, %buffer1, %iv1) : (i128, memref<2xi128>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first1 = "memref.load"(%buffer1, %zero) : (memref<2xi128>, index) -> i128
    %updated1 = "arith.addi"(%first1, %seed1) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    "memref.store"(%updated1, %buffer1, %zero) : (i128, memref<2xi128>, index) -> ()
    %value1 = "memref.load"(%buffer1, %zero) : (memref<2xi128>, index) -> i128
    %dim1 = "memref.dim"(%buffer1, %zero) : (memref<2xi128>, index) -> index
    %observed1 = "func.call"(%value1, %caller) <{callee = @touch}> : (i128, memref<?xi128>) -> i128
    "memref.dealloc"(%buffer1) : (memref<2xi128>) -> ()
    %result = "arith.addi"(%observed0, %observed1) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    "func.return"(%result, %dim1) : (i128, index) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "touch", function_type = (i128, memref<?xi128>) -> i128}> ({
  ^bb0(%value: i128, %caller: memref<?xi128>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %old = "memref.load"(%caller, %zero) : (memref<?xi128>, index) -> i128
    %new = "arith.addi"(%old, %value) <{overflowFlags = #arith.overflow<none>}> : (i128, i128) -> i128
    "memref.store"(%new, %caller, %zero) : (i128, memref<?xi128>, index) -> ()
    "func.return"(%new) : (i128) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%input: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%input, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
}) : () -> ()