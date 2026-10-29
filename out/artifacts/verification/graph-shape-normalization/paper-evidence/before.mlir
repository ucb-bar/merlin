builtin.module {
 func.func @probe(%a: memref<2x3xi32>, %b: memref<2x3xi32>, %sink: memref<1xi32>) -> (index, index, i32) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %axis = arith.constant 1 : index
  %other = arith.constant 0 : index
  %first = "memref.cast"(%a) : (memref<2x3xi32>) -> memref<2x3xi32>
  %second = "memref.cast"(%first) : (memref<2x3xi32>) -> memref<2x3xi32>
  %d0 = memref.dim %second, %axis : memref<2x3xi32>
  %d1 = memref.dim %b, %other : memref<2x3xi32>
  %value = memref.load %second[%zero, %zero] : memref<2x3xi32>
  memref.store %value, %sink[%zero] : memref<1xi32>
  func.return %d0, %d1, %value : index, index, i32
 }
 func.func @untouched(%value: i32) -> i32 {
  func.return %value : i32
 }
}
