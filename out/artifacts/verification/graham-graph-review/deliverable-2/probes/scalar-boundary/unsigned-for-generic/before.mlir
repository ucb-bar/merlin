builtin.module {
  func.func @k(%a:i8, %b:i8) -> i8 {
    %l0 = arith.constant 0 : index
    %s = arith.constant 1 : index
    "scf.for"(%l0, %l0, %s) <{unsignedCmp}> ({
    ^bb1(%i: index):
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    func.return %z : i8
  }
}