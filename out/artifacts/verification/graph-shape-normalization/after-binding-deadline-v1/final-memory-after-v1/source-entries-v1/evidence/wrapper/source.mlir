"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, memref<1xi32>) -> (i32, index)}> ({
  ^bb0(%seed0: i32, %seed1: i32, %caller: memref<1xi32>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %one = "arith.constant"() <{value = 1 : index}> : () -> index
    %bound0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %bound1 = "arith.constant"() <{value = 3 : index}> : () -> index
    %buffer0 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3xi32>
    %view0 = "memref.cast"(%buffer0) : (memref<2x3xi32>) -> memref<2x3xi32>
    %again0 = "memref.cast"(%view0) : (memref<2x3xi32>) -> memref<2x3xi32>
    %rows0 = "memref.dim"(%again0, %zero) : (memref<2x3xi32>, index) -> index
    %columns0 = "memref.dim"(%again0, %one) : (memref<2x3xi32>, index) -> index
    "scf.for"(%zero, %bound0, %one) ({
    ^bb0(%iv0: index):
      "scf.for"(%zero, %bound1, %one) ({
      ^bb0(%iv0_1: index):
        "memref.store"(%seed0, %again0, %iv0, %iv0_1) : (i32, memref<2x3xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first0 = "memref.load"(%again0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %updated0 = "arith.addi"(%first0, %seed0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%updated0, %again0, %zero, %zero) : (i32, memref<2x3xi32>, index, index) -> ()
    %value0 = "memref.load"(%again0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %dim0 = "memref.dim"(%again0, %zero) : (memref<2x3xi32>, index) -> index
    %observed0 = "func.call"(%value0, %caller) <{callee = @touch}> : (i32, memref<1xi32>) -> i32
    "memref.dealloc"(%buffer0) : (memref<2x3xi32>) -> ()
    %buffer1 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3xi32>
    %view1 = "memref.cast"(%buffer1) : (memref<2x3xi32>) -> memref<2x3xi32>
    %again1 = "memref.cast"(%view1) : (memref<2x3xi32>) -> memref<2x3xi32>
    %rows1 = "memref.dim"(%again1, %zero) : (memref<2x3xi32>, index) -> index
    %columns1 = "memref.dim"(%again1, %one) : (memref<2x3xi32>, index) -> index
    "scf.for"(%zero, %bound0, %one) ({
    ^bb0(%iv1: index):
      "scf.for"(%zero, %bound1, %one) ({
      ^bb0(%iv1_1: index):
        "memref.store"(%seed1, %again1, %iv1, %iv1_1) : (i32, memref<2x3xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first1 = "memref.load"(%again1, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %updated1 = "arith.addi"(%first1, %seed1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%updated1, %again1, %zero, %zero) : (i32, memref<2x3xi32>, index, index) -> ()
    %value1 = "memref.load"(%again1, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %dim1 = "memref.dim"(%again1, %zero) : (memref<2x3xi32>, index) -> index
    %observed1 = "func.call"(%value1, %caller) <{callee = @touch}> : (i32, memref<1xi32>) -> i32
    "memref.dealloc"(%buffer1) : (memref<2x3xi32>) -> ()
    %result = "arith.addi"(%observed0, %observed1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result, %dim1) : (i32, index) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "touch", function_type = (i32, memref<1xi32>) -> i32}> ({
  ^bb0(%value: i32, %caller: memref<1xi32>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %old = "memref.load"(%caller, %zero) : (memref<1xi32>, index) -> i32
    %new = "arith.addi"(%old, %value) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%new, %caller, %zero) : (i32, memref<1xi32>, index) -> ()
    "func.return"(%new) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i32) -> i32}> ({
  ^bb0(%input: i32):
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %result = "arith.addi"(%input, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%result) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "wrapper", function_type = (i32, i32, memref<1xi32>) -> (i32, index)}> ({
  ^bb0(%a: i32, %b: i32, %caller: memref<1xi32>):
    %x, %y = "func.call"(%a, %b, %caller) <{callee = @kernel}> : (i32, i32, memref<1xi32>) -> (i32, index)
    "func.return"(%x, %y) : (i32, index) -> ()
  }) : () -> ()
}) : () -> ()