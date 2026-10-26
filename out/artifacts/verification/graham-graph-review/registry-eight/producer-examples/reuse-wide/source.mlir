builtin.module {
 func.func @kernel(%seed0: i128, %seed1: i128, %caller: memref<?xi128>) -> (i128, index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %bound0 = arith.constant 2 : index
  %buffer0 = memref.alloc() {alignment = 16 : i64} : memref<2xi128>
  scf.for %iv0_0 = %zero to %bound0 step %one {
   memref.store %seed0, %buffer0[%iv0_0] : memref<2xi128>
   scf.yield
  }
  %first0 = memref.load %buffer0[%zero] : memref<2xi128>
  %updated0 = arith.addi %first0, %seed0 : i128
  memref.store %updated0, %buffer0[%zero] : memref<2xi128>
  %value0 = memref.load %buffer0[%zero] : memref<2xi128>
  %dim0 = memref.dim %buffer0, %zero : memref<2xi128>
  %observed0 = func.call @touch(%value0, %caller) : (i128, memref<?xi128>) -> i128
  memref.dealloc %buffer0 : memref<2xi128>
  %buffer1 = memref.alloc() {alignment = 16 : i64} : memref<2xi128>
  scf.for %iv1_0 = %zero to %bound0 step %one {
   memref.store %seed1, %buffer1[%iv1_0] : memref<2xi128>
   scf.yield
  }
  %first1 = memref.load %buffer1[%zero] : memref<2xi128>
  %updated1 = arith.addi %first1, %seed1 : i128
  memref.store %updated1, %buffer1[%zero] : memref<2xi128>
  %value1 = memref.load %buffer1[%zero] : memref<2xi128>
  %dim1 = memref.dim %buffer1, %zero : memref<2xi128>
  %observed1 = func.call @touch(%value1, %caller) : (i128, memref<?xi128>) -> i128
  memref.dealloc %buffer1 : memref<2xi128>
  %result = arith.addi %observed0, %observed1 : i128
  func.return %result, %dim1 : i128, index
 }
 func.func @touch(%value: i128, %caller: memref<?xi128>) -> i128 {
  %zero = arith.constant 0 : index
  %old = memref.load %caller[%zero] : memref<?xi128>
  %new = arith.addi %old, %value : i128
  memref.store %new, %caller[%zero] : memref<?xi128>
  func.return %new : i128
 }
 func.func @untouched(%input: i32) -> i32 {
  %one = arith.constant 1 : i32
  %result = arith.addi %input, %one : i32
  func.return %result : i32
 }
}

