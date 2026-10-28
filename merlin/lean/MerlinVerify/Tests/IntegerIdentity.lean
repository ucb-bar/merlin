/- Seeded attacks at the untrusted SSA interface, plus poison-preserving actual execution. -/
import MerlinVerify.Check.Module
import MerlinVerify.Rules.IntegerIdentity
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IntegerIdentity

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Rules.IntegerIdentity

def ty : Ty := .int 13

def literal (k : Kind) (id : Nat) : Op :=
  mkOp "arith.constant" [v id ty] [] [("value", .int (k.neutral 13).toInt ty)]

def site (k : Kind) (rev : Bool) : List Op :=
  [literal k 1, mkOp k.name [v 2 ty] (ordered rev 0 1)]

def mapping : Rho := [(0, 100)]

#guard candidates.all fun (k, rev) =>
  (Rules.IntegerIdentity.check mapping (site k rev) []).isSome == decide (k.reverseOK rev)
#guard candidates.all fun (k, rev) =>
  (Rules.IntegerIdentity.check mapping (site k rev) [literal k 101]).isSome == decide (k.reverseOK rev)

-- A target constant cannot replace the target argument that implements the input.
#guard (Rules.IntegerIdentity.check mapping (site .add false) [literal .add 100]).isNone
-- The source constant may not shadow that input before its actual arithmetic use.
#guard (Rules.IntegerIdentity.check mapping
  [literal .add 0, mkOp "arith.addi" [v 2 ty] [0, 0]] []).isNone
-- The runtime input must have an actual correspondence; a type annotation does not supply it.
#guard (Rules.IntegerIdentity.check mapping
  [literal .add 1, mkOp "arith.addi" [v 2 ty] [99, 1]] []).isNone
-- A changed neutral value and an unproved overflow attribute are independently refused.
#guard (Rules.IntegerIdentity.check mapping
  [literal .mul 1, mkOp "arith.addi" [v 2 ty] [0, 1]] []).isNone
#guard (Rules.IntegerIdentity.check mapping
  [literal .add 1, mkOp "arith.addi" [v 2 ty] [0, 1]
    [("overflowFlags", .enum "arith" "overflow" "nsw")]] []).isNone

def source : Module :=
  { ops := [func "f" [v 0 ty, v 3 ty] (site .add false ++ [ret [2, 3]])] }

def target : Module :=
  { ops := [func "f" [v 100 ty, v 103 ty] [ret [100, 103]]] }

def wrongTarget : Module :=
  { ops := [func "f" [v 100 ty, v 103 ty] [ret [103, 103]]] }

#guard moduleCheck source target
#guard !moduleCheck source wrongTarget

def returnsPoison (m : Module) : Bool :=
  match Sem.Ops.denote Rules.IntegerFold.reference m "f"
      [.scalar .poison, .scalar (.some (.bits 13 9))] with
  | .ok (.scalar .poison :: _) => true
  | _ => false

#guard returnsPoison source && returnsPoison target

end MerlinVerify.Tests.IntegerIdentity
