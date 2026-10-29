"builtin.module"() ({
  "func.func"() <{sym_name = "k", function_type = (i8, i8) -> (i8, tensor<2xi8>)}> ({
  ^bb0(%a: i8, %b: i8):
    %c = "arith.constant"() <{value = dense_resource<blob1> : tensor<2xi8>}> : () -> tensor<2xi8>
    %x = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %z = "arith.muli"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    "func.return"(%z, %c) : (i8, tensor<2xi8>) -> ()
  }) : () -> ()
}) : () -> ()