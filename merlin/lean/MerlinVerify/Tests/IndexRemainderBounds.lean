/- Actual-width positive divisors justify unsigned remainder bounds, including high-bit inputs. -/
import MerlinVerify.Check.IndexRemainderBounds

namespace MerlinVerify.Tests.IndexRemainderBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness
open IndexRanges IndexRemainderBounds

def indexFact (info : Option Interval := none) : IndexRanges.Fact := ⟨⟨.scalar .index,true⟩,info⟩
def remainder (name : String := "arith.remui") (attrs : List (String × Attr) := []) : Op :=
  .mk name [⟨3,.index⟩] [1,2] attrs [] [] []
def output (ib : Nat) (r : Option Interval) := rule ib [] (remainder) [indexFact,indexFact r]
def info (rs : Option (List IndexRanges.Fact)) := rs.bind (fun fs => fs.head?.map (·.info))

#guard divisor? 32 (some ⟨3,4⟩) == some ⟨3,4⟩
#guard divisor? 64 (some ⟨1,8⟩) == some ⟨1,8⟩
#guard divisor? 32 (some ⟨0,8⟩) == none
#guard divisor? 32 none == none
#guard divisor? 0 (some ⟨1,2⟩) == none
#guard divisor? 1 (some ⟨1,2⟩) == none
#guard divisor? 2 (some ⟨1,2⟩) == some ⟨1,2⟩
#guard divisor? 32 (some ⟨2147483647,2147483648⟩) == some ⟨2147483647,2147483648⟩
#guard divisor? 32 (some ⟨2147483648,2147483649⟩) == none
#guard divisor? 64 (some ⟨2147483648,2147483649⟩) == some ⟨2147483648,2147483649⟩
#guard info (output 32 (some ⟨3,4⟩)) == some (some ⟨0,3⟩)
#guard info (output 64 (some ⟨1,8⟩)) == some (some ⟨0,7⟩)
#guard (output 32 none).isNone
#guard (output 32 (some ⟨0,1⟩)).isNone
#guard (rule 32 [] (remainder) [⟨⟨.scalar .index,false⟩,none⟩,indexFact (some ⟨3,4⟩)]).isNone
#guard (rule 32 [] (remainder) [indexFact,⟨⟨.scalar .index,false⟩,some ⟨3,4⟩⟩]).isNone
#guard (rule 32 [] (remainder) [⟨⟨.scalar (.int 32),true⟩,none⟩,indexFact (some ⟨3,4⟩)]).isNone
#guard (rule 32 [] (remainder "arith.remsi") [indexFact,indexFact (some ⟨3,4⟩)]).isNone
#guard (rule 32 [] (remainder "arith.divui") [indexFact,indexFact (some ⟨3,4⟩)]).isNone
#guard ["nonNeg","isExact","disjoint","unknown.meaning"].all (fun attr =>
  (rule 32 [] (remainder "arith.remui" [(attr,.unit)]) [indexFact,indexFact (some ⟨3,4⟩)]).isNone)
#guard (rule 32 [] (remainder "arith.remui" [("overflowFlags",.enum "arith" "overflow" "nuw")])
  [indexFact,indexFact (some ⟨3,4⟩)]).isNone
#guard (rule 32 [] (remainder "arith.remui" [("fastmath",.enum "arith" "fastmath" "nnan")])
  [indexFact,indexFact (some ⟨3,4⟩)]).isNone

-- All112 four-bit input/positive-low-divisor pairs preserve nonnegative signed coordinates.
#guard (List.range 16).all (fun x => (List.range 7).all (fun d =>
  let result := (BitVec.ofNat 4 x) % (BitVec.ofNat 4 (d+1))
  decide (0 ≤ result.toInt ∧ result.toInt < d+1)))

end MerlinVerify.Tests.IndexRemainderBounds
