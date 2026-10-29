"builtin.module"() ({
  "func.func"() <{sym_name = "run", function_type = (i32) -> ()}> ({
  ^bb0(%input: i32):
    %slot0 = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %slot1 = "arith.constant"() <{value = 1 : i8}> : () -> i8
    %slot2 = "arith.constant"() <{value = 2 : i8}> : () -> i8
    %slot3 = "arith.constant"() <{value = 3 : i8}> : () -> i8
    %value0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %value3 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %value5 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %value7 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %value4294967295 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %equal3 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    "configuration_fixture_isa.work"(%slot0, %value0, %value0) : (i8, i32, i32) -> ()
    "configuration_fixture_isa.pair"(%slot0, %value3, %value5) : (i8, i32, i32) -> ()
    "configuration_fixture_isa.work"(%slot0, %value0, %value0) : (i8, i32, i32) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()