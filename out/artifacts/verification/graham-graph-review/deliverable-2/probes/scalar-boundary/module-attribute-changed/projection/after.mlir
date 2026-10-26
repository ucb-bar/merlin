builtin.module attributes {x.a = 2 : i64} {
  func.func @k(%a:i8, %b:i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %z = arith.muli %x, %x : i8
    func.return %z : i8
  }
}