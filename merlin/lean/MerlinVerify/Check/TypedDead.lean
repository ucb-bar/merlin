/- Reusable exact deletion of one total typed producer. Each operation family establishes
actual successful execution separately; freshness and alpha checking protect all continuation
uses, including captured values and preexisting bindings with the same numeric identifier. -/
import MerlinVerify.Check.SplatNormalize

namespace MerlinVerify.Check.TypedDead

open MerlinVerify MerlinVerify.Sem.Ops

def unused (Γ : TEnv) (s : Op) (tail : List Op) : Bool :=
  freshIds (SplatNormalize.diagonal Γ) (s.results.map (·.id)) &&
    alphaOps (SplatNormalize.diagonal Γ) tail tail

/-- This helper cannot remove a potentially failing operation: the caller must first supply
its actual totality witness for the current typed environment. -/
theorem drop_sound {Γ : TEnv} {s : Op} {tail : List Op} {ctx : Ctx} {call : CallSem} {env : Env}
    (hu : unused Γ s tail = true)
    (ht : ∃ r value, s.results = [r] ∧
      stepOp opSem ctx call env s = .ok ((r.id, value) :: env)) :
    ∃ env', stepOp opSem ctx call env s = .ok env' ∧
      evalOps opSem ctx call env' tail = evalOps opSem ctx call env tail := by
  simp only [unused, Bool.and_eq_true] at hu
  obtain ⟨hfresh, htail⟩ := hu
  obtain ⟨r, value, hr, hstep⟩ := ht
  have hf : (SplatNormalize.diagonal Γ).notImage r.id :=
    (freshIds_sound hfresh).2 r.id (by simp [hr])
  have hc := Rules.DequantSite.Corr.temporary (SplatNormalize.diagonal_refl Γ env) hf value
  exact ⟨_, hstep, (alphaOps_sound call call _ tail tail ctx _ _ htail hc (fun _ _ => rfl)).symm⟩

end MerlinVerify.Check.TypedDead
