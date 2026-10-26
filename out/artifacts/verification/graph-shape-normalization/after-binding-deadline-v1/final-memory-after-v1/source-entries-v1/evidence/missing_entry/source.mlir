builtin.module {
 func.func @kernel(%seed0: i32, %seed1: i32, %caller: memref<1xi32>) -> (i32, index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %bound0 = arith.constant 2 : index
  %bound1 = arith.constant 3 : index
  %buffer0 = memref.alloc() {alignment = 16 : i64} : memref<2x3xi32>
  %view0 = "memref.cast"(%buffer0) : (memref<2x3xi32>) -> memref<2x3xi32>
  %again0 = "memref.cast"(%view0) : (memref<2x3xi32>) -> memref<2x3xi32>
  %rows0 = memref.dim %again0, %zero : memref<2x3xi32>
  %columns0 = memref.dim %again0, %one : memref<2x3xi32>
  scf.for %iv0_0 = %zero to %bound0 step %one {
   scf.for %iv0_1 = %zero to %bound1 step %one {
    memref.store %seed0, %again0[%iv0_0, %iv0_1] : memref<2x3xi32>
    scf.yield
   }
   scf.yield
  }
  %first0 = memref.load %again0[%zero, %zero] : memref<2x3xi32>
  %updated0 = arith.addi %first0, %seed0 : i32
  memref.store %updated0, %again0[%zero, %zero] : memref<2x3xi32>
  %value0 = memref.load %again0[%zero, %zero] : memref<2x3xi32>
  %dim0 = memref.dim %again0, %zero : memref<2x3xi32>
  %observed0 = func.call @touch(%value0, %caller) : (i32, memref<1xi32>) -> i32
  memref.dealloc %buffer0 : memref<2x3xi32>
  %buffer1 = memref.alloc() {alignment = 16 : i64} : memref<2x3xi32>
  %view1 = "memref.cast"(%buffer1) : (memref<2x3xi32>) -> memref<2x3xi32>
  %again1 = "memref.cast"(%view1) : (memref<2x3xi32>) -> memref<2x3xi32>
  %rows1 = memref.dim %again1, %zero : memref<2x3xi32>
  %columns1 = memref.dim %again1, %one : memref<2x3xi32>
  scf.for %iv1_0 = %zero to %bound0 step %one {
   scf.for %iv1_1 = %zero to %bound1 step %one {
    memref.store %seed1, %again1[%iv1_0, %iv1_1] : memref<2x3xi32>
    scf.yield
   }
   scf.yield
  }
  %first1 = memref.load %again1[%zero, %zero] : memref<2x3xi32>
  %updated1 = arith.addi %first1, %seed1 : i32
  memref.store %updated1, %again1[%zero, %zero] : memref<2x3xi32>
  %value1 = memref.load %again1[%zero, %zero] : memref<2x3xi32>
  %dim1 = memref.dim %again1, %zero : memref<2x3xi32>
  %observed1 = func.call @touch(%value1, %caller) : (i32, memref<1xi32>) -> i32
  memref.dealloc %buffer1 : memref<2x3xi32>
  %result = arith.addi %observed0, %observed1 : i32
  func.return %result, %dim1 : i32, index
 }
 func.func @touch(%value: i32, %caller: memref<1xi32>) -> i32 {
  %zero = arith.constant 0 : index
  %old = memref.load %caller[%zero] : memref<1xi32>
  %new = arith.addi %old, %value : i32
  memref.store %new, %caller[%zero] : memref<1xi32>
  func.return %new : i32
 }
 func.func @untouched(%input: i32) -> i32 {
  %one = arith.constant 1 : i32
  %result = arith.addi %input, %one : i32
  func.return %result : i32
 }
}
