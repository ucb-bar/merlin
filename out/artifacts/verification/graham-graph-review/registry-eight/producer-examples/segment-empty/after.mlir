"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lower = "arith.constant"() <{value = 0 : index}> : () -> index
    %upper = "arith.constant"() <{value = 0 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 6 : index}> : () -> index
    %r_1 = "scf.for"(%lower, %upper, %r, %a) ({
    ^bb1(%0: index, %1: i8):
      %2 = "arith.addi"(%0, %r) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %upper) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %step, %1) ({
      ^bb2(%5: index, %6: i8):
        %invariant = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%invariant, %5) : (i8, index) -> ()
        %next = "arith.subi"(%6, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "scf.yield"(%next) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%4) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%r_1) : (i8) -> ()
  }) : () -> ()
}) : () -> ()