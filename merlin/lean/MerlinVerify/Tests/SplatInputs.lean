/- Geometry obligations for uniform input compaction, including loss/change of loop extents. -/
import MerlinVerify.Rules.SplatInputs

namespace MerlinVerify.Tests.SplatInputs

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg
open MerlinVerify.Rules.SplatInputs

def i32 : Ty := .int 32
def parallel (n : Nat) : List IterKind := List.replicate n .parallel
def emptyMap (n : Nat) : AffineMap := { numDims := n, numSyms := 0, results := [] }

#guard checkLayout [.splat [2, 3] i32 (identityMap 2)] [identityMap 2]
  (parallel 2) [.tensor [2, 3] i32] == some [2, 3]
#guard checkLayout [.keep (.tensor [2, 3] i32) (identityMap 2),
    .splat [3] i32 { numDims := 2, numSyms := 0, results := [.dim 1] }]
  [identityMap 2] (parallel 2) [.tensor [2, 3] i32] == some [2, 3]
#guard checkLayout [.splat [2, 3] i32 (identityMap 2),
    .splat [3, 2] (.float .f32) { numDims := 2, numSyms := 0, results := [.dim 1, .dim 0] }]
  [identityMap 2, identityMap 2] (parallel 2)
  [.tensor [2, 3] i32, .tensor [2, 3] (.float .f32)] == some [2, 3]
#guard checkLayout [.splat [] i32 (emptyMap 0)] [identityMap 0]
  [] [.tensor [] i32] == some []
#guard checkLayout [.splat [0, 3] i32 (identityMap 2)] [identityMap 2]
  (parallel 2) [.tensor [0, 3] i32] == some [0, 3]
#guard checkLayout [.splat [2] i32
    { numDims := 2, numSyms := 0, results := [.floordiv (.dim 0) (.const 2)] }]
  [identityMap 2] (parallel 2) [.tensor [4, 3] i32] == some [4, 3]

-- The semantic kernel also preserves reductions in their original order. The production pass
-- currently selects parallel generics only; this stronger lemma grants no reordering permission.
#guard checkLayout [.keep (.tensor [2] i32) (identityMap 1), .splat [2] i32 (identityMap 1)]
  [emptyMap 1] [.reduction] [.tensor [] i32] == some [2]

-- Both layouts are individually valid here, but compaction changes the first determining extent.
#guard (structuredRanges? [identityMap 1, identityMap 1] 1 [.tensor [2] i32, .tensor [3] i32]).isSome
#guard (structuredRanges? [emptyMap 1, identityMap 1] 1 [.tensor [] i32, .tensor [3] i32]).isSome
#guard checkLayout [.splat [2] i32 (identityMap 1)] [identityMap 1]
  (parallel 1) [.tensor [3] i32] == none

-- No remaining operand determines the extent after compaction.
#guard checkLayout [.splat [2] i32 (identityMap 1)] [emptyMap 1]
  (parallel 1) [.tensor [] i32] == none
#guard checkLayout [.splat [2] i32 (emptyMap 1)] [identityMap 1]
  (parallel 1) [.tensor [2] i32] == none
#guard checkLayout [.splat [2] i32 { numDims := 1, numSyms := 0, results := [.const 2] }]
  [identityMap 1] (parallel 1) [.tensor [2] i32] == none
#guard checkLayout [.splat [2] i32 { numDims := 1, numSyms := 1, results := [.sym 0] }]
  [identityMap 1] (parallel 1) [.tensor [2] i32] == none
#guard checkLayout [.splat [2] i32 (identityMap 1)] []
  (parallel 1) [.tensor [2] i32] == none

-- Initialization is deliberately absent from the theorem's payload domain: uniform poison must
-- be preserved, rather than turned into a defined constant by compaction.
example : RVal.HasTy 64
    ((Slot.splat [2] i32 (identityMap 1)).value true (.scalar .poison)) (.tensor [] i32) :=
  value_typed true _ (by trivial)

end MerlinVerify.Tests.SplatInputs
