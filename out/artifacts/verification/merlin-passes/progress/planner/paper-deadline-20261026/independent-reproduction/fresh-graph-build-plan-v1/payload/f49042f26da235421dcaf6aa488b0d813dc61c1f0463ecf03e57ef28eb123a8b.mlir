builtin.module {
  func.func private @observe(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    %r = func.call @observe(%z) : (i8) -> i8
    func.return %r : i8
  }
  func.func @other(%a: i8) -> i8 {
    func.return %a : i8
  }
}