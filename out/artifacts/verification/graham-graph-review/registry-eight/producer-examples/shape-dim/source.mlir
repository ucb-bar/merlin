builtin.module {
func.func @kernel(%a:memref<4xi8>) -> index {
  %z = arith.constant 0 : index
  %r = memref.dim %a, %z : memref<4xi8>
  func.return %r : index
}}
