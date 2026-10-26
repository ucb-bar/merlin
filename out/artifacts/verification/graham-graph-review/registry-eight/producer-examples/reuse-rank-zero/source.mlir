builtin.module {
func.func @kernel(%seed:i32) -> i32 {
 %a = memref.alloc() : memref<i32>
 memref.store %seed, %a[] : memref<i32>
 %x = memref.load %a[] : memref<i32>
 memref.dealloc %a : memref<i32>
 %b = memref.alloc() : memref<i32>
 memref.store %seed, %b[] : memref<i32>
 %y = memref.load %b[] : memref<i32>
 memref.dealloc %b : memref<i32>
 %r = arith.addi %x, %y : i32
 func.return %r : i32
}}
