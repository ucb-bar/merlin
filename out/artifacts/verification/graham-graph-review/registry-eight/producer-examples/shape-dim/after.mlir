"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (memref<4xi8>) -> index}> ({
  ^bb0(%a: memref<4xi8>):
    %z = "arith.constant"() <{value = 0 : index}> : () -> index
    %r = "arith.constant"() <{value = 4 : index}> : () -> index
    "func.return"(%r) : (index) -> ()
  }) : () -> ()
}) : () -> ()