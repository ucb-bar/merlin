"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lower = "arith.constant"() <{value = 0 : index}> : () -> index
    %upper = "arith.constant"() <{value = 7 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %invariant = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
    %r = "scf.for"(%lower, %upper, %step, %a) ({
    ^bb1(%i: index, %acc: i8):
      "opaque.observe"(%invariant, %i) : (i8, index) -> ()
      %next = "arith.subi"(%acc, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
      "scf.yield"(%next) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%r) : (i8) -> ()
  }) : () -> ()
}) : () -> ()