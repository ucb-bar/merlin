/- Checked bitwise singleton bounds, initialization-only unknowns and promise refusal. -/
import MerlinVerify.Tests.IndexArithmetic

namespace MerlinVerify.Tests.IndexBitwiseBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness

#guard IndexArithmetic.accepts 64 "arith.andi" 3 1 2
#guard IndexArithmetic.accepts 32 "arith.ori" 0 1 2
#guard IndexArithmetic.accepts 64 "arith.xori" 3 1 3
#guard IndexArithmetic.accepts 64 "arith.andi" 0 0 1
#guard !IndexArithmetic.accepts 64 "arith.andi" 0 0 0
#guard !IndexArithmetic.accepts 32 "arith.ori" 2 1 3
#guard IndexArithmetic.accepts 32 "arith.andi" 4294967296 4294967296 1
#guard !IndexArithmetic.accepts 64 "arith.andi" 4294967296 4294967296 1
#guard IndexArithmetic.accepts 32 "arith.ori" 4294967296 1 2
#guard !IndexArithmetic.accepts 64 "arith.ori" 4294967296 1 2
#guard IndexArithmetic.accepts 64 "arith.xori" 3 1 3 1
#guard !IndexArithmetic.accepts 64 "arith.xori" 3 1 3 2

def unknown (name : String) (extract : Bool) : Module :=
  { ops := [func "f" [v 0 (tensorTy [2] i32), v 1 .index, v 2 .index]
    ([mkOp name [v 3 .index] [1, 2]] ++
     (if extract then [mkOp "tensor.extract" [v 4 i32] [0, 3], ret [4]] else [ret [3]]))] }

#guard ([32, 64].all fun ib => ["arith.andi", "arith.ori", "arith.xori"].all fun name =>
  Bounds.check ib (unknown name false) "f" [.tensor [2] i32, .scalar .index, .scalar .index] [.scalar .index] &&
  !Bounds.check ib (unknown name true) "f" [.tensor [2] i32, .scalar .index, .scalar .index] [.scalar i32])

#guard (["arith.andi", "arith.ori", "arith.xori"].all fun name =>
  [[("disjoint", Attr.unit)], [("fastmath", Attr.enum "arith" "fastmath" "nnan")],
   [("nonNeg", Attr.unit)], [("unreviewed.meaning", Attr.unit)]].all fun flags =>
    !Bounds.check 64 (IndexArithmetic.source name 0 1 2 flags) "f"
      [.tensor [2] i32, .scalar .index] [.scalar i32])

-- Both input constants must have genuine nonnegative singleton evidence. Exhaust small signed
-- widths too, so a negative input never acquires a bound just because a bit mask could clear it.
#guard ([2, 3, 32, 64].all fun ib =>
  ["arith.andi", "arith.ori", "arith.xori"].all fun name =>
  (List.range 4).all fun a => (List.range 4).all fun b => (List.range 5).all fun size =>
    let lhs := (BitVec.ofNat ib a).toInt
    let rhs := (BitVec.ofNat ib b).toInt
    let result := (match name with
      | "arith.andi" => BitVec.ofNat ib a &&& BitVec.ofNat ib b
      | "arith.ori" => BitVec.ofNat ib a ||| BitVec.ofNat ib b
      | _ => BitVec.ofNat ib a ^^^ BitVec.ofNat ib b).toInt
    IndexArithmetic.accepts ib name a b size == decide (0 ≤ lhs ∧ 0 ≤ rhs ∧ 0 ≤ result ∧ result < (size : Int)))

end MerlinVerify.Tests.IndexBitwiseBounds
