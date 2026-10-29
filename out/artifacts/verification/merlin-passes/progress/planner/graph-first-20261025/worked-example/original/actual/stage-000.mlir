"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lower = "arith.constant"() <{value = 0 : index}> : () -> index
    %upper = "arith.constant"() <{value = 7 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %result = "scf.for"(%lower, %upper, %step, %a) ({
    ^bb0(%i: index, %acc: i8):
      %first = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      %product = "arith.muli"(%first, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      "opaque.observe"(%first, %i) : (i8, index) -> ()
      %next = "arith.subi"(%acc, %product) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      "scf.yield"(%next) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%result) : (i8) -> ()
  }) : () -> ()
}) : () -> ()