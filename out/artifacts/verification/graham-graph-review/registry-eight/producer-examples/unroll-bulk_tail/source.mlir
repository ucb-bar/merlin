builtin.module {
 func.func @kernel(%input0: i32, %input1: i32, %capture: i32, %memory: memref<?xi32>) -> (i32, i32) {
  %lower = arith.constant 3 : index
  %upper = arith.constant 25 : index
  %step = arith.constant 2 : index
  %result:2 = scf.for %iv = %lower to %upper step %step iter_args(%carry0 = %input0, %carry1 = %input1) -> (i32, i32) {
   %next0 = arith.addi %carry0, %capture : i32
   %next1 = arith.addi %carry1, %next0 : i32
   memref.store %next0, %memory[%iv] : memref<?xi32>
   scf.yield %next0, %next1 : i32, i32
  }
  %observer = arith.addi %result#0, %capture : i32
  func.return %observer, %result#1 : i32, i32
 }
 func.func @untouched(%value: i32) -> i32 {
  %one = arith.constant 1 : i32
  %result = arith.addi %value, %one : i32
  func.return %result : i32
 }
}

