"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i32) -> i32}> ({
  ^bb0(%seed: i32):
    %a = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%seed, %a) : (i32, memref<i32>) -> ()
    %x = "memref.load"(%a) : (memref<i32>) -> i32
    "memref.store"(%seed, %a) : (i32, memref<i32>) -> ()
    %y = "memref.load"(%a) : (memref<i32>) -> i32
    "memref.dealloc"(%a) : (memref<i32>) -> ()
    %r = "arith.addi"(%x, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) : () -> ()
}) : () -> ()