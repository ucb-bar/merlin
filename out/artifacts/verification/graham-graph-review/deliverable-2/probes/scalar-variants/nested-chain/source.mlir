builtin.module {
  func.func @kernel(%a:i8,%b:i8) -> i8 {
    %l=arith.constant 0 : index
    %u=arith.constant 3 : index
    %s=arith.constant 1 : index
    %outer=scf.for %i=%l to %u step %s iter_args(%acc=%a) -> (i8) {
      %inner=scf.for %j=%l to %u step %s iter_args(%state=%acc) -> (i8) {
        %invariant=arith.muli %a,%b : i8
        %duplicate=arith.muli %a,%b : i8
        "opaque.observe"(%i,%j,%invariant) : (index,index,i8) -> ()
        %next=arith.addi %state,%duplicate : i8
        scf.yield %next : i8
      }
      scf.yield %inner : i8
    }
    func.return %outer : i8
  }
}