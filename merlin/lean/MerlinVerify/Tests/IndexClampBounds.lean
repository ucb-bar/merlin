/- Unsigned-minimum bounds use actual nonnegative operand facts and the actual signed limit. -/
import MerlinVerify.Check.IndexClampBounds

namespace MerlinVerify.Tests.IndexClampBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness
open IndexRanges IndexClampBounds

def indexFact (info : Option Interval := none) : IndexRanges.Fact := ⟨⟨.scalar .index,true⟩,info⟩
def minimum (attrs : List (String × Attr) := []) : Op :=
  .mk "arith.minui" [⟨3,.index⟩] [1,2] attrs [] [] []
def output (ib : Nat) (a b : Option Interval) := rule ib [] (minimum) [indexFact a,indexFact b]

#guard evidence 32 none (some ⟨7,8⟩) == some ⟨0,8⟩
#guard evidence 64 (some ⟨7,8⟩) none == some ⟨0,8⟩
#guard evidence 32 (some ⟨0,8⟩) (some ⟨0,4⟩) == some ⟨0,4⟩
#guard evidence 32 none none == none
#guard evidence 0 none (some ⟨0,1⟩) == none
#guard evidence 1 none (some ⟨0,1⟩) == some ⟨0,1⟩
#guard evidence 1 none (some ⟨1,2⟩) == none
#guard evidence 32 none (some ⟨2147483647,2147483648⟩) == some ⟨0,2147483648⟩
#guard evidence 32 none (some ⟨2147483648,2147483649⟩) == none
#guard evidence 64 none (some ⟨2147483648,2147483649⟩) == some ⟨0,2147483649⟩
#guard (bits (BitVec.ofInt 32 (-1)) (BitVec.ofNat 32 7)).toNat == 7
#guard (bits (BitVec.ofNat 64 7) (BitVec.ofInt 64 (-1))).toInt == 7
#guard ((output 32 none (some ⟨7,8⟩)).bind (·.head?)).map (fun f => (f.base.initialized,f.info)) == some (true,some ⟨0,8⟩)
#guard ((output 64 none none).bind (·.head?)).map (fun f => (f.base.initialized,f.info)) == some (true,none)
#guard (rule 32 [] (minimum) [⟨⟨.scalar .index,false⟩,none⟩,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum) [⟨⟨.scalar (.int 32),true⟩,none⟩,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum [("nonNeg",.unit)]) [indexFact,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum [("isExact",.unit)]) [indexFact,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum [("disjoint",.unit)]) [indexFact,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum [("overflowFlags",.enum "arith" "overflow" "nsw")])
  [indexFact,indexFact (some ⟨7,8⟩)]).isNone
#guard (rule 32 [] (minimum [("unknown.meaning",.unit)]) [indexFact,indexFact (some ⟨7,8⟩)]).isNone

-- All 128 four-bit unsigned-input/low-cap combinations retain nonnegative read coordinates.
#guard (List.range 16).all (fun x => (List.range 8).all (fun cap =>
  let result := bits (BitVec.ofNat 4 x) (BitVec.ofNat 4 cap)
  decide (0 ≤ result.toInt ∧ result.toInt ≤ cap)))

end MerlinVerify.Tests.IndexClampBounds
