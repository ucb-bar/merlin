builtin.module {
  func.func @k(%a:i8, %b:i8) -> i8 {
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    "x.region"() ({
    }) : () -> ()
    func.return %z : i8
  }
}