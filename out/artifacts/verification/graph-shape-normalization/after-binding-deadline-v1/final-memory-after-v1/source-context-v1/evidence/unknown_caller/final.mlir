"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32, i32, memref<1xi32>) -> (i32, index)}> ({
  ^bb0(%seed0: i32, %seed1: i32, %caller: memref<1xi32>):
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %one = "arith.constant"() <{value = 1 : index}> : () -> index
    %bound0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %bound1 = "arith.constant"() <{value = 3 : index}> : () -> index
    %buffer0 = "memref.alloc"() <{alignment = 16 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3xi32>
    %rows0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %columns0 = "arith.constant"() <{value = 3 : index}> : () -> index
    %0 = "arith.constant"() <{value = 2 : index}> : () -> index
    "scf.for"(%zero, %bound0, %0) ({
    ^bb0(%1: index):
      %2 = "arith.addi"(%1, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %bound0) : (index, index) -> index
      "scf.for"(%1, %3, %one) ({
      ^bb0(%4: index):
        "scf.for"(%zero, %bound1, %one) ({
        ^bb0(%iv0: index):
          "memref.store"(%seed0, %buffer0, %4, %iv0) : (i32, memref<2x3xi32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first0 = "memref.load"(%buffer0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %updated0 = "arith.addi"(%first0, %seed0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%updated0, %buffer0, %zero, %zero) : (i32, memref<2x3xi32>, index, index) -> ()
    %value0 = "memref.load"(%buffer0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %dim0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %observed0 = "func.call"(%value0, %caller) <{callee = @touch}> : (i32, memref<1xi32>) -> i32
    %rows1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %columns1 = "arith.constant"() <{value = 3 : index}> : () -> index
    %5 = "arith.constant"() <{value = 2 : index}> : () -> index
    "scf.for"(%zero, %bound0, %5) ({
    ^bb0(%6: index):
      %7 = "arith.addi"(%6, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %8 = "arith.minui"(%7, %bound0) : (index, index) -> index
      "scf.for"(%6, %8, %one) ({
      ^bb0(%9: index):
        "scf.for"(%zero, %bound1, %one) ({
        ^bb0(%iv1: index):
          "memref.store"(%seed1, %buffer0, %9, %iv1) : (i32, memref<2x3xi32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %first1 = "memref.load"(%buffer0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %updated1 = "arith.addi"(%first1, %seed1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%updated1, %buffer0, %zero, %zero) : (i32, memref<2x3xi32>, index, index) -> ()
    %value1 = "memref.load"(%buffer0, %zero, %zero) : (memref<2x3xi32>, index, index) -> i32
    %dim1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %observed1 = "func.call"(%value1, %caller) <{callee = @touch}> : (i32, memref<1xi32>) -> i32
    "memref.dealloc"(%buffer0) : (memref<2x3xi32>) -> ()
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
  "func.func"() <{sym_name = "unknown", function_type = (i32, i32, memref<1xi32>) -> (i32, index), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "wrapper", function_type = (i32, i32, memref<1xi32>, i32) -> (i32, index)}> ({
  ^bb0(%a: i32, %b: i32, %caller: memref<1xi32>, %bias: i32):
    %value, %extent = "func.call"(%a, %b, %caller) <{callee = @unknown}> : (i32, i32, memref<1xi32>) -> (i32, index)
    %continued = "arith.addi"(%value, %bias) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %zero = "arith.constant"() <{value = 0 : index}> : () -> index
    %old = "memref.load"(%caller, %zero) : (memref<1xi32>, index) -> i32
    %observed = "arith.addi"(%old, %continued) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%observed, %caller, %zero) : (i32, memref<1xi32>, index) -> ()
    "func.return"(%observed, %extent) : (i32, index) -> ()
  }) : () -> ()
}) : () -> ()