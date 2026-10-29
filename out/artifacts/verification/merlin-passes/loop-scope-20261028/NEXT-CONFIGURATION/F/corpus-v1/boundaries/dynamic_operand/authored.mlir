module {
func.func @run(%input: i32) {
  %slot0 = arith.constant 0 : i8
  %slot1 = arith.constant 1 : i8
  %slot2 = arith.constant 2 : i8
  %slot3 = arith.constant 3 : i8
  %value0 = arith.constant 0 : i32
  %value3 = arith.constant 3 : i32
  %value5 = arith.constant 5 : i32
  %value7 = arith.constant 7 : i32
  %value4294967295 = arith.constant 4294967295 : i32
  %equal3 = arith.constant 3 : i32
  "configuration_fixture_isa.pair"(%slot0, %value3, %value5) : (i8, i32, i32) -> ()
  "configuration_fixture_isa.set"(%slot0, %input, %value0) : (i8, i32, i32) -> ()
  "configuration_fixture_isa.pair"(%slot0, %value3, %value5) : (i8, i32, i32) -> ()
  "configuration_fixture_isa.pair"(%slot0, %value3, %value5) : (i8, i32, i32) -> ()
  func.return
}
}
