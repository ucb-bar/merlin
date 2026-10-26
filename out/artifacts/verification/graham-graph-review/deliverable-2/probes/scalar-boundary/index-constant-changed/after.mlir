builtin.module {
  func.func @k(%a:i8, %b:i8) -> i8 {
    %l = arith.constant 1 : index
    "x.obs"(%l) : (index) -> ()
    %x = arith.addi %a, %b : i8
    %z = arith.muli %x, %x : i8
    func.return %z : i8
  }
}