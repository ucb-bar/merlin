builtin.module {
  func.func private @k(%a:i8, %b:i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %z = arith.muli %x, %x : i8
    func.return %z : i8
  }
}