builtin.module {
  func.func @callee(%v:i8) -> i8 {
    func.return %v : i8
  }
  func.func @k(%a:i8, %b:i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %z = arith.muli %x, %x : i8
    %r = func.call @callee(%z) : (i8) -> i8
    func.return %r : i8
  }
}