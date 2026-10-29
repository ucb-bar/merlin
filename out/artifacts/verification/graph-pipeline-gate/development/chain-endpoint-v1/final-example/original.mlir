builtin.module {
  func.func private @observe(i8) -> i8
  func.func @main(%a: i8, %b: i8) -> i8 {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %st = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %st iter_args(%c = %a) -> (i8) {
      %x = arith.addi %a, %b : i8
      %y = arith.addi %a, %b : i8
      %z = arith.muli %x, %y : i8
      %v = arith.addi %c, %z : i8
      %o = func.call @observe(%v) : (i8) -> i8
      scf.yield %o : i8
    }
    %out = func.call @observe(%r) : (i8) -> i8
    func.return %out : i8
  }
  func.func @other(%a: i8) -> i8 {
    func.return %a : i8
  }
}