builtin.module {
func.func @kernel(%a:memref<4xi8>) -> memref<4xi8> {
  %r = "memref.cast"(%a) : (memref<4xi8>) -> memref<4xi8>
  func.return %r : memref<4xi8>
}}
