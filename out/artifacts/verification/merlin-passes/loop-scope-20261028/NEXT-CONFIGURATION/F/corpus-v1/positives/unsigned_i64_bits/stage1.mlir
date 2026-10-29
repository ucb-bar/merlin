"builtin.module"() ({
  "func.func"() <{sym_name = "run", function_type = (i64) -> ()}> ({
  ^bb0(%input: i64):
    %slot0 = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %slot1 = "arith.constant"() <{value = 1 : i8}> : () -> i8
    %slot2 = "arith.constant"() <{value = 2 : i8}> : () -> i8
    %slot3 = "arith.constant"() <{value = 3 : i8}> : () -> i8
    %value0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %value3 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %value5 = "arith.constant"() <{value = 5 : i64}> : () -> i64
    %value7 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    %value18446744073709551615 = "arith.constant"() <{value = -1 : i64}> : () -> i64
    %equal3 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    "configuration_fixture_isa.pair"(%slot0, %value3, %value5) : (i8, i64, i64) -> ()
    "configuration_fixture_isa.set"(%slot2, %value18446744073709551615, %value0) : (i8, i64, i64) -> ()
    "configuration_fixture_isa.work"(%slot0, %value0, %value0) : (i8, i64, i64) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()