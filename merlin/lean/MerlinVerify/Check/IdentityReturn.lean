/- Compare a whole identity-generic return block with an argument-only return block.
This is a refinement comparator, never an exact tensor-alias normalization. -/
import MerlinVerify.Check.IdentityReturnProof

namespace MerlinVerify.Check.IdentityReturn

open MerlinVerify MerlinVerify.Sem.Ops
open IdentityReturnSyntax IdentityReturnProof

def check (ib : Nat) (Γ : TEnv) (ρ : Rho) : List Op → List Op → Bool
  | [s, sourceRet], [targetRet] =>
    match candidate ib s with
    | some g =>
      decide (Legal ib Γ s g) && decide (DefaultTerm "func.return" sourceRet) &&
        decide (DefaultTerm "func.return" targetRet) && sourceRet.operands.contains g.result.id &&
        (sourceRet.operands.mapM (returnId Γ ρ g.result.id g.input) == some targetRet.operands)
    | none => false
  | _, _ => false

/-- Acceptance gives strict whole-block refinement for actual typed inputs and the actual
argument correspondence. Arbitrary call interpretations cannot observe the canonical/raw
representation difference, since both accepted blocks end directly with func.return. -/
theorem check_sound {ib : Nat} {Γ : TEnv} {ρ : Rho} {ss ts : List Op}
    (h : check ib Γ ρ ss ts = true) (ctx : Ctx) (callS callT : CallSem) (envS envT : Env)
    (he : EnvTyped ib Γ envS) (hc : Corr ρ envS envT) (_hi : ctx.indexBits = ib) :
    Refines ResultsRefine (evalOps opSem ctx callT envT ts) (evalOps opSem ctx callS envS ss) := by
  unfold check at h
  split at h
  · next s sourceRet targetRet =>
    cases hg : candidate ib s with
    | none => simp [hg] at h
    | some g =>
      simp only [hg, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
      obtain ⟨⟨⟨⟨hl, hs⟩, ht⟩, _⟩, hids⟩ := h
      exact block_refines hl (candidate_regions hg) hs ht hids ctx callS callT envS envT he hc
  · cases h

end MerlinVerify.Check.IdentityReturn
