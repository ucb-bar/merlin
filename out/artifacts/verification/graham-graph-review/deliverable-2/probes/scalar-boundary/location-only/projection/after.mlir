builtin.module {
  func.func @k(%a:i8, %b:i8) -> i8 {
    %x = arith.addi %a, %b : i8 loc("x.py":1:1)
    %z = arith.muli %x, %x : i8
    func.return %z : i8
  }
}