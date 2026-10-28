/- Exclusive endpoint and signed-limit controls for intervals learned from actual input bits. -/
import MerlinVerify.Check.DefinedBounds
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.IndexRangeAdd

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics
open MerlinVerify.Check.Definedness

def source (width size : Nat) (offset : Int) (extract := true) : Module :=
  { ops := [func "f" [v 0 (tensorTy [size] i32), v 1 (.int width)]
    ([mkOp "arith.index_castui" [v 2 .index] [1],
      mkOp "arith.constant" [v 3 .index] [] [("value", .int offset .index)],
      mkOp "arith.addi" [v 4 .index] [2, 3]] ++
     (if extract then [mkOp "tensor.extract" [v 5 i32] [0, 4], ret [5]] else [ret [4]]))] }

def accepts (ib width size : Nat) (offset : Int) : Bool :=
  Bounds.check ib (source width size offset) "f" [.tensor [size] i32, .scalar (.int width)] [.scalar i32]

#guard accepts 32 8 257 1
#guard !accepts 32 8 256 1
#guard accepts 64 1 2 0
#guard !accepts 64 1 1 0
#guard !accepts 64 1 0 0
#guard !accepts 32 8 256 (-1)
#guard accepts 32 31 2147483648 0
#guard !accepts 32 31 2147483649 1
#guard accepts 64 31 2147483649 1
#guard accepts 32 1 2147483648 2147483646
#guard !accepts 32 1 2147483649 2147483647
#guard Bounds.check 32 (source 31 2 1 false) "f"
  [.tensor [2] i32, .scalar (.int 31)] [.scalar .index]

#guard IndexRangeAdd.evidence 32 (some ⟨1, 1⟩) (some ⟨0, 1⟩) == none
#guard IndexRangeAdd.evidence 32 (some ⟨0, 2⟩) none == none
#guard IndexRangeAdd.evidence 3 (some ⟨0, 2⟩) (some ⟨1, 3⟩) == some ⟨1, 4⟩
#guard IndexRangeAdd.evidence 3 (some ⟨0, 2⟩) (some ⟨1, 4⟩) == none

-- Exhaust interval endpoints for widths2/3. Check every actual bitvector pair in each accepted
-- interval against independently decoded signed addition; no wrapped negative result is admitted.
#guard ([2, 3].all fun ib => (List.range 5).all fun lo1 => (List.range 5).all fun hi1 =>
  (List.range 5).all fun lo2 => (List.range 5).all fun hi2 =>
    match IndexRangeAdd.evidence ib (some ⟨lo1, hi1⟩) (some ⟨lo2, hi2⟩) with
    | none => true
    | some result =>
      (List.range (2 ^ ib)).all fun x => (List.range (2 ^ ib)).all fun y =>
        let a := (BitVec.ofNat ib x).toInt
        let b := (BitVec.ofNat ib y).toInt
        if (lo1 : Int) ≤ a ∧ a < hi1 ∧ (lo2 : Int) ≤ b ∧ b < hi2 then
          let actual := (BitVec.ofNat ib x + BitVec.ofNat ib y).toInt
          decide ((result.lower : Int) ≤ actual ∧ actual < result.upper)
        else true)

end MerlinVerify.Tests.IndexRangeAdd
