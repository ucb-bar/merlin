/- Integer commutation is exact; subtraction, floating arithmetic, and changed flags are refused. -/
import MerlinVerify.Check.Module
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.Commutative

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def moduleOf (name : String) (flipped : Bool) (ty : Ty := i32)
    (attrs : List (String × Attr) := []) : Module :=
  { ops := [func "f" [v 0 ty, v 1 ty]
    [mkOp name [v 2 ty] (if flipped then [1, 0] else [0, 1]) attrs, ret [2]]] }

#guard Rules.Commutative.names.all fun name => moduleCheck (moduleOf name false) (moduleOf name true)
#guard moduleCheck (moduleOf "arith.addi" false) (moduleOf "arith.addi" false)
#guard moduleCheck (moduleOf "arith.muli" false (tensorTy [2, 3] i32))
  (moduleOf "arith.muli" true (tensorTy [2, 3] i32))
#guard !moduleCheck (moduleOf "arith.subi" false) (moduleOf "arith.subi" true)
#guard !moduleCheck (moduleOf "arith.divsi" false) (moduleOf "arith.divsi" true)
#guard !moduleCheck (moduleOf "arith.addf" false (.float .f32)) (moduleOf "arith.addf" true (.float .f32))
#guard !moduleCheck (moduleOf "arith.addi" false i32 [("overflowFlags", .enum "arith" "overflow" "nsw")])
  (moduleOf "arith.addi" true i32 [("overflowFlags", .enum "arith" "overflow" "nsw")])
#guard !moduleCheck (moduleOf "arith.addi" false)
  (moduleOf "arith.addi" true i32 [("overflowFlags", .enum "arith" "overflow" "nsw")])

-- Poison propagation and mismatched shape/width behavior are included in the universal theorem.
example (w : Nat) (x : BitVec w) :
    Sem.Ops.Arith.intBin (fun w x y => .some (.bits w (x + y))) .poison (.some (.bits w x)) =
    Sem.Ops.Arith.intBin (fun w x y => .some (.bits w (x + y))) (.some (.bits w x)) .poison := rfl

end MerlinVerify.Tests.Commutative
