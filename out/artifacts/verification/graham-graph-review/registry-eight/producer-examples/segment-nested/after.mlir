"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %lower = "arith.constant"() <{value = 0 : index}> : () -> index
    %upper = "arith.constant"() <{value = 7 : index}> : () -> index
    %step = "arith.constant"() <{value = 2 : index}> : () -> index
    %r = "arith.constant"() <{value = 6 : index}> : () -> index
    %r_1 = "scf.for"(%lower, %upper, %r, %a) ({
    ^bb1(%0: index, %1: i8):
      %2 = "arith.addi"(%0, %r) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %3 = "arith.minui"(%2, %upper) : (index, index) -> index
      %4 = "scf.for"(%0, %3, %step, %1) ({
      ^bb2(%5: index, %6: i8):
        %inner = "arith.constant"() <{value = 4 : index}> : () -> index
        %inner_1 = "scf.for"(%lower, %upper, %inner, %6) ({
        ^bb3(%7: index, %8: i8):
          %9 = "arith.addi"(%7, %inner) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %10 = "arith.minui"(%9, %upper) : (index, index) -> index
          %11 = "scf.for"(%7, %10, %step, %8) ({
          ^bb4(%12: index, %13: i8):
            %invariant = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
            "opaque.observe"(%invariant, %5, %12) : (i8, index, index) -> ()
            %next = "arith.subi"(%13, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
            "scf.yield"(%next) : (i8) -> ()
          }) : (index, index, index, i8) -> i8
          "scf.yield"(%11) : (i8) -> ()
        }) : (index, index, index, i8) -> i8
        "scf.yield"(%inner_1) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%4) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%r_1) : (i8) -> ()
  }) : () -> ()
}) : () -> ()