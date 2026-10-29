"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lower = "arith.constant"() <{value = 0 : index}> : () -> index
    %upper = "arith.constant"() <{value = 7 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 4 : index}> : () -> index
    %r_1 = "arith.constant"() <{value = 8 : index}> : () -> index
    %r_2 = "scf.for"(%lower, %r_1, %r, %a) ({
    ^bb1(%0: index, %1: i8):
      %inner = "arith.constant"() <{value = 4 : index}> : () -> index
      %inner_1 = "arith.constant"() <{value = 8 : index}> : () -> index
      %inner_2 = "scf.for"(%lower, %inner_1, %inner, %1) ({
      ^bb2(%2: index, %3: i8):
        %invariant = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%invariant, %0, %2) : (i8, index, index) -> ()
        %next = "arith.subi"(%3, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %4 = "arith.constant"() <{value = 2 : index}> : () -> index
        %5 = "arith.addi"(%2, %4) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %invariant_1 = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%invariant_1, %0, %5) : (i8, index, index) -> ()
        %next_1 = "arith.subi"(%next, %invariant_1) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "scf.yield"(%next_1) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      %6 = "arith.constant"() <{value = 2 : index}> : () -> index
      %7 = "arith.addi"(%0, %6) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %inner_3 = "arith.constant"() <{value = 4 : index}> : () -> index
      %inner_4 = "arith.constant"() <{value = 8 : index}> : () -> index
      %inner_5 = "scf.for"(%lower, %inner_4, %inner_3, %inner_2) ({
      ^bb3(%8: index, %9: i8):
        %invariant_2 = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%invariant_2, %7, %8) : (i8, index, index) -> ()
        %next_2 = "arith.subi"(%9, %invariant_2) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        %10 = "arith.constant"() <{value = 2 : index}> : () -> index
        %11 = "arith.addi"(%8, %10) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %invariant_3 = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%invariant_3, %7, %11) : (i8, index, index) -> ()
        %next_3 = "arith.subi"(%next_2, %invariant_3) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "scf.yield"(%next_3) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%inner_5) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%r_2) : (i8) -> ()
  }) : () -> ()
}) : () -> ()