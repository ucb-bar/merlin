/- Qualification at the untrusted JSON/SSA boundary, including identifiers that a valid MLIR
printer would never emit. A malicious fragment must not overwrite a prior correspondence. -/
import MerlinVerify.Rules.IntegerFold
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerFold

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def literal (id : Nat) (value : Int) : Op :=
  mkOp "arith.constant" [v id i32] [] [("value", .int value i32)]

def fragment : List Op :=
  [literal 0 7, literal 1 3, mkOp "arith.addi" [v 2 i32] [0, 1]]

#guard (Rules.IntegerFold.check [] fragment [literal 10 10]).isSome
#guard (Rules.IntegerFold.check [] fragment [literal 10 11]).isNone

-- The source's local constant 0 may not hide an input that already has a correspondence.
#guard (Rules.IntegerFold.check [(0, 100)] fragment [literal 10 10]).isNone
-- The target may not overwrite a target input that still implements an unrelated source input.
#guard (Rules.IntegerFold.check [(50, 10)] fragment [literal 10 10]).isNone

-- An external SSA input is not a closed constant, even if its type is a fixed-width integer.
#guard (Rules.IntegerFold.check []
  [literal 0 7, mkOp "arith.addi" [v 2 i32] [0, 99]] [literal 10 10]).isNone

-- Repeated source identifiers use the newest actual value; stale bindings cannot justify a fold.
#guard (Rules.IntegerFold.check []
  [literal 0 7, literal 0 8, literal 1 3, mkOp "arith.addi" [v 2 i32] [0, 1]]
  [literal 10 10]).isNone
#guard (Rules.IntegerFold.check []
  [literal 0 7, literal 0 8, literal 1 3, mkOp "arith.addi" [v 2 i32] [0, 1]]
  [literal 10 11]).isSome

end MerlinVerify.Tests.IntegerFold
