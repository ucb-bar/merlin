/- Actual runtime bits, checked scalar index inputs, exact predicates and target bool values. -/
import MerlinVerify.Rules.IndexCompare
import MerlinVerify.Tests.IndexFold

namespace MerlinVerify.Tests.IndexCompare

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules.IndexCompare
open MerlinVerify.Tests.IndexFold (constant facts bits)

def Γ : TEnv := [(1,.scalar .index),(2,.scalar .index)]
def comparison (p : Int) (x : Nat := 1) (y : Nat := 2) (ty : Ty := .int 1)
    (extra : List (String × Attr) := []) : Op := .mk "arith.cmpi" [⟨3,ty⟩] [x,y]
  ([("predicate",.int p (.int 64))] ++ extra) [] [] []
def boolean (value : Bool) : Op := .mk "arith.constant" [⟨9,.int 1⟩] [] [("value",.bool value)] [] [] []

#guard (value? 32 Γ (facts 32 17 5) (comparison 0)).isSome
#guard bits (value? 32 Γ (facts 32 17 5) (comparison 2)) == some 0
#guard bits (value? 64 Γ (facts 64 17 5) (comparison 4)) == some 1
#guard (replacement? 32 Γ (facts 32 17 5) [boolean false] (comparison 2)).isSome
#guard (replacement? 32 Γ (facts 32 17 5) [boolean true] (comparison 2)).isNone
#guard (value? 32 [] (facts 32 17 5) (comparison 2)).isNone
#guard (value? 32 [(1,.scalar (.int 32)),(2,.scalar (.int 32))] (facts 32 17 5) (comparison 2)).isNone
#guard (value? 32 Γ [] (comparison 2)).isNone
#guard (value? 32 Γ (facts 32 17 5) (comparison 10)).isNone
#guard (value? 32 Γ (facts 32 17 5) (comparison (-1))).isNone
#guard (value? 32 Γ (facts 32 17 5) (comparison 2 (ty := .index))).isNone
#guard (value? 32 Γ (facts 32 17 5) (comparison 2 (extra := [("nonNeg",.unit)]))).isNone
#guard (value? 32 Γ (facts 32 17 5) (comparison 2 (extra := [("unknown.meaning",.unit)]))).isNone
#guard (replacement? 32 Γ [⟨1,.scalar .poison⟩,⟨2,.scalar (.some (.bits 32 0))⟩]
  [boolean false] (comparison 2)).isNone

-- S6: the same candidate is correct at64 and incorrect at32, including actual wrapped arithmetic.
def wrap ib := MerlinVerify.Rules.IndexFold.advance ib (facts ib 2147483647 1)
  (MerlinVerify.Tests.IndexFold.binary "arith.addi" 1 2 4)
def withZero ib := MerlinVerify.Rules.IndexFold.advance ib (wrap ib) (constant 5 0)
def wrapΓ : TEnv := [(4,.scalar .index),(5,.scalar .index)]
def negative : Op := comparison 2 4 5
#guard bits (value? 32 wrapΓ (withZero 32) negative) == some 1
#guard bits (value? 64 wrapΓ (withZero 64) negative) == some 0
#guard (replacement? 32 wrapΓ (withZero 32) [boolean false] negative).isNone
#guard (replacement? 64 wrapΓ (withZero 64) [boolean false] negative).isSome
#guard (replacement? 32 wrapΓ (withZero 32) [boolean true] negative).isSome
#guard (MerlinVerify.Rules.IndexFold.lookup (family.advance 32 Γ (facts 32 17 5) (comparison 2)) 3).isNone

end MerlinVerify.Tests.IndexCompare
