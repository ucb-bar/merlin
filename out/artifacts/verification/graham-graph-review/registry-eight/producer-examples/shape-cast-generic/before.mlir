"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (memref<4xi8>) -> memref<4xi8>}> ({
  ^bb0(%a: memref<4xi8>):
    %r = "memref.cast"(%a) : (memref<4xi8>) -> memref<4xi8>
    "func.return"(%r) : (memref<4xi8>) -> ()
  }) : () -> ()
}) : () -> ()