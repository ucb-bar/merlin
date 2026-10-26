builtin.module {
  func.func private @observe(i8) -> i8
  func.func private @other(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.muli %a, %b : i8
    %first = func.call @observe(%x) : (i8) -> i8
    %z = arith.subi %a, %b : i8
    %second = func.call @other(%y) : (i8) -> i8
    %r = arith.addi %first, %second : i8
    %out = arith.addi %r, %z : i8
    func.return %out : i8
  }
  func.func @untouched(%a: i8) -> i8 { func.return %a : i8 }
}