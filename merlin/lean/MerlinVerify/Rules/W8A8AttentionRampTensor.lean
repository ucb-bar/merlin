/- The exact actual initialized i64 arange tensor. Both independently produced
ramps therefore contain identical words at a matching diagonal coordinate. -/
import MerlinVerify.Check.PolicyW8A8AttentionRamp
import MerlinVerify.Rules.W8A8SourcePointwise

namespace MerlinVerify.Rules.W8A8AttentionRampTensor

open MerlinVerify Sem.Ops.Linalg

def ramp (extent : Nat) : Tensor :=
  Tensor.canon [extent] (.int 64) (fun point => W8A8AttentionMaskBody.integer (BitVec.ofNat 64 (point.getD 0 0)))

theorem generic_sound {actual : Region} (accepted : Check.PolicyW8A8AttentionRamp.check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (extent : Nat) (initial : Idx → Val)
    (width : ctx.indexBits = 64)
    (geometry : Check.StructuredOk [identityMap 1] 1 [.tensor [extent] (.int 64)] [extent]) :
    genericSem ctx [identityMap 1] [.parallel] [] [.tensor ⟨[extent], .int 64, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) = .ok [.tensor (ramp extent)] := by
  apply W8A8SourcePointwise.generic_sound W8A8SourceSemantics.opSem ctx call env []
    ⟨[extent], .int 64, initial⟩ _ [.parallel] actual geometry (by rfl) _
  intro point bounds old
  cases point with
  | nil => simp [Idx.inBounds_cons_nil] at bounds
  | cons column rest =>
    have bounded := (Idx.inBounds_cons_cons ..).mp bounds
    have empty : rest = [] := (Idx.inBounds_nil_iff _).mp bounded.2
    subst rest
    have executed := Check.PolicyW8A8AttentionRamp.check_sound accepted { ctx with loopIndex := [column] }
      call env column old width rfl
    simpa [W8A8TensorOperands.values, ramp] using executed

theorem matching_words (extent : Nat) (column : Nat) (bounded : column < extent) :
    (ramp extent).get [column] = W8A8AttentionMaskBody.integer (BitVec.ofNat 64 column) := by
  have bounds : Idx.InBounds [extent] [column] :=
    (Idx.inBounds_cons_cons ..).mpr ⟨bounded, (Idx.inBounds_nil_iff _).mpr rfl⟩
  simp [ramp, Tensor.canon, bounds]

end MerlinVerify.Rules.W8A8AttentionRampTensor
