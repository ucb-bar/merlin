"builtin.module"() ({
  "func.func"() <{sym_name = "main", function_type = (index) -> index}> ({
  ^bb0(%x: index):
    %z = "arith.constant"() <{value = 0 : index}> : () -> index
    %v = "arith.addi"(%x, %z) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "func.return"(%v) : (index) -> ()
  }) : () -> ()
}) : () -> ()