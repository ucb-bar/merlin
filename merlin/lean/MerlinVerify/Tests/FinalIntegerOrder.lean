/- One final ordering-only traversal follows all aliases, including captured parent renames. -/
import MerlinVerify.Check.IntegerBody
import MerlinVerify.Rules.CastCancel
import MerlinVerify.Tests.Semantics

namespace MerlinVerify.Tests.FinalIntegerOrder

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def Γ : TEnv := [(4, .scalar (.int 8)), (5, .scalar (.int 8))]
def cast (kind : IntegerCast.Kind) (x r width : Nat) : Op :=
  .mk kind.name [⟨r, .int width⟩] [x] [] [] [] []
def add (x y r : Nat) : Op := .mk "arith.addi" [⟨r, .int 8⟩] [x, y] [] [] [] []
def yield (x : Nat) : Op := .mk "linalg.yield" [] [x] [] [] [] []
def source (kind : IntegerCast.Kind) : List Op :=
  [cast kind 4 6 32, cast .trunc 6 7 8, add 7 5 8, yield 8]
def target : List Op := [add 4 5 6, yield 6]

def early (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential (IntegerBody.base ib) (CastCancel.base ib)
def final (ib : Nat) : SplatNormalizer.Normalizer ib :=
  BodyNormalize.sequential
    (BodyNormalize.sequential (IntegerBody.unorderedBase ib) (CastCancel.base ib))
    (IntegerBody.integerOrder ib)
def normalized (base : SplatNormalizer.Normalizer ib) (ops : List Op) :=
  base.run 0 Γ [] ops [.scalar (.int 8)]
def aligned (base : SplatNormalizer.Normalizer ib) (kind : IntegerCast.Kind) : Bool :=
  match normalized base (source kind), normalized base target with
  | some src, some tgt => alphaOps (SplatNormalize.diagonal Γ) src tgt
  | _, _ => false

#guard ((normalized (early 64) (source .zero)).bind (·[0]?)).map Op.operands == some [5, 4]
#guard ((normalized (early 64) target).bind (·[0]?)).map Op.operands == some [4, 5]
#guard !(aligned (early 64) .zero)
#guard !(aligned (early 32) .sign)
#guard aligned (final 64) .zero
#guard aligned (final 32) .zero
#guard aligned (final 64) .sign
#guard aligned (final 32) .sign
#guard ((normalized (final 64) (source .zero)).bind (·[0]?)).map Op.operands == some [4, 5]

open MerlinVerify.Tests.Semantics

def shaped : Ty := tensorTy [3] (.int 8)
def capturedΓ : TEnv :=
  [(0, .scalar (.int 8)), (1, .scalar (.int 8)),
   (2, .tensor [3] (.int 8)), (3, .tensor [3] (.int 8))]
def capturedGeneric (operand : Nat) : Op :=
  mkOp "linalg.generic" [v 6 shaped] [2, 3]
    [("indexing_maps", .array [amap 1 [0], amap 1 [0]]),
     ("iterator_types", .array [iter "parallel"]),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [.mk [.mk [v 7 (.int 8), v 8 (.int 8)]
      [add operand 1 9, add 7 9 10, mkOp "arith.xori" [v 11 (.int 8)] [10, 8], yield 11]]]
def capturedSource : List Op := [cast .sign 0 4 32, cast .trunc 4 5 8, capturedGeneric 5, ret [6]]
def capturedTarget : List Op := [capturedGeneric 0, ret [6]]
def capturedAligned (src tgt : Option (List Op)) : Bool :=
  match src, tgt with
  | some src, some tgt => alphaOps (SplatNormalize.diagonal capturedΓ) src tgt
  | _, _ => false
def localFinal (ops : List Op) :=
  (BodyNormalize.recursive (final 64) 1).run 0 capturedΓ [] ops [.tensor [3] (.int 8)]
def wholeFinal (ib : Nat) (ops : List Op) :=
  (CastCancel.normalizer.sequential Check.IntegerBody.ordering).run
    ib 0 capturedΓ ops [] [.tensor [3] (.int 8)]

-- A parent alias changes two captured IDs after child-first local ordering has finished.
#guard !(capturedAligned (localFinal capturedSource) (localFinal capturedTarget))
#guard capturedAligned (wholeFinal 32 capturedSource) (wholeFinal 32 capturedTarget)
#guard capturedAligned (wholeFinal 64 capturedSource) (wholeFinal 64 capturedTarget)

end MerlinVerify.Tests.FinalIntegerOrder
