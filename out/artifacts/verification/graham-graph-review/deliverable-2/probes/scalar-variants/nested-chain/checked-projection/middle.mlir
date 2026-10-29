"builtin.module"() ({
  "func.func"() <{sym_name = "kernel", function_type = (i8, i8) -> i8}> ({
  ^bb0(%a: i8, %b: i8):
    %l = "arith.constant"() <{value = 0 : index}> : () -> index
    %u = "arith.constant"() <{value = 3 : index}> : () -> index
    %s = "arith.constant"() <{value = 1 : index}> : () -> index
    %outer = "scf.for"(%l, %u, %s, %a) ({
    ^bb1(%i: index, %acc: i8):
      %inner = "scf.for"(%l, %u, %s, %acc) ({
      ^bb2(%j: index, %state: i8):
        %invariant = "arith.muli"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "opaque.observe"(%i, %j, %invariant) : (index, index, i8) -> ()
        %next = "arith.addi"(%state, %invariant) <{overflowFlags = #arith.overflow<none>}> : (i8, i8) -> i8
        "scf.yield"(%next) : (i8) -> ()
      }) : (index, index, index, i8) -> i8
      "scf.yield"(%inner) : (i8) -> ()
    }) : (index, index, index, i8) -> i8
    "func.return"(%outer) : (i8) -> ()
  }) : () -> ()
}) : () -> ()