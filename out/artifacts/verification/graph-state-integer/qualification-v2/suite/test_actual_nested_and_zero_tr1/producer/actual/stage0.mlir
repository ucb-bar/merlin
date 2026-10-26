"builtin.module"() ({
  "func.func"() <{sym_name = "provider", function_type = (i16, i16, index) -> (i16, i16), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "watch", function_type = (i16, i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "other", function_type = (i16, i16) -> (), sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{sym_name = "main", function_type = (i16, i16) -> (i16, i16)}> ({
  ^bb0(%x: i16, %bias: i16):
    %lb = "arith.constant"() <{value = 0 : index}> : () -> index
    %ub = "arith.constant"() <{value = 0 : index}> : () -> index
    %innerub = "arith.constant"() <{value = 2 : index}> : () -> index
    %st = "arith.constant"() <{value = 1 : index}> : () -> index
    %zero = "arith.constant"() <{value = 0 : i16}> : () -> i16
    %one = "arith.constant"() <{value = 1 : i16}> : () -> i16
    %a = "arith.constant"() <{value = -3 : i16}> : () -> i16
    %b = "arith.constant"() <{value = 3 : i16}> : () -> i16
    %fold = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
    %out, %out_1 = "scf.for"(%lb, %ub, %st, %x, %bias) ({
    ^bb0(%iv: index, %first: i16, %second: i16):
      %neutral = "arith.addi"(%first, %fold) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
      %inner, %inner_1 = "scf.for"(%lb, %innerub, %st, %neutral, %second) ({
      ^bb0(%j: index, %left: i16, %right: i16):
        %capture = "arith.muli"(%bias, %one) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        %keep = "arith.subi"(%left, %zero) <{overflowFlags = #arith.overflow<none>}> : (i16, i16) -> i16
        "func.call"(%keep, %capture) <{callee = @watch}> : (i16, i16) -> ()
        %v, %v_1 = "func.call"(%keep, %right, %j) <{callee = @provider}> : (i16, i16, index) -> (i16, i16)
        "scf.yield"(%v, %v_1) : (i16, i16) -> ()
      }) : (index, index, index, i16, i16) -> (i16, i16)
      "scf.yield"(%inner, %inner_1) : (i16, i16) -> ()
    }) : (index, index, index, i16, i16) -> (i16, i16)
    "func.call"(%out, %out_1) <{callee = @watch}> : (i16, i16) -> ()
    "func.return"(%out_1, %out) : (i16, i16) -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "untouched", function_type = (i16) -> i16}> ({
  ^bb0(%x: i16):
    "func.return"(%x) : (i16) -> ()
  }) : () -> ()
}) : () -> ()