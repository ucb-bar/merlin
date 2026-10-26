builtin.module {
  func.func @kernel(%upper:index, %a:i8, %b:i8) -> i8 {
    %lower = arith.constant 0 : index
    
    %step = arith.constant 2 : index
    %r = scf.for %i = %lower to %upper step %step iter_args(%acc=%a) -> (i8) {
      %invariant = arith.addi %a, %b : i8
      "opaque.observe"(%invariant, %i) : (i8,index) -> ()
      %next = arith.subi %acc, %invariant : i8
      scf.yield %next : i8
    }
    func.return %r : i8
  }
}
