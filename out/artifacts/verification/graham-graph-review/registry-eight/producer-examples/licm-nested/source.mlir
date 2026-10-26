builtin.module {
  func.func @kernel(%a:i8, %b:i8) -> i8 {
    %lower = arith.constant 0 : index
    %upper = arith.constant 7 : index
    %step = arith.constant 2 : index
    %r = scf.for %i = %lower to %upper step %step iter_args(%acc=%a) -> (i8) {
      %inner = scf.for %j = %lower to %upper step %step iter_args(%state=%acc) -> (i8) {
        %invariant = arith.addi %a, %b : i8
        "opaque.observe"(%invariant, %i, %j) : (i8,index,index) -> ()
        %next = arith.subi %state, %invariant : i8
        scf.yield %next : i8
      }
      scf.yield %inner : i8
    }
    func.return %r : i8
  }
}
