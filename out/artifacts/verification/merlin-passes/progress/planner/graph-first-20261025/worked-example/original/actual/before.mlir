builtin.module {
  func.func @kernel(%a:i8, %b:i8) -> (i8) {
    %lower = arith.constant 0 : index
    %upper = arith.constant 7 : index
    %step = arith.constant 2 : index
    %result = scf.for %i = %lower to %upper step %step iter_args(%acc=%a) -> (i8) {
      %first = arith.addi %a, %b : i8
      %dup = arith.addi %a, %b : i8
    %dead = arith.subi %a, %b : i8
      %product = arith.muli %dup, %b : i8
      "opaque.observe"(%first, %i) : (i8,index) -> ()
      %next = arith.subi %acc, %product : i8
      
      scf.yield %next : i8
    }
    func.return %result : i8
  }
}