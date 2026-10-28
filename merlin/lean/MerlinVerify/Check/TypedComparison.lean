/- Typed comparisons of actual normal forms. Each alternative carries its own refinement proof;
the caller supplies the original runtime domain and checked SSA correspondence. -/
import MerlinVerify.Check.TypedC1

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

structure TypedComparison where
  run : Nat → TEnv → Rho → List Op → List Op → Bool
  sound : ∀ ib Γ ρ ss ts ctx callS callT envS envT,
    run ib Γ ρ ss ts = true → EnvTyped ib Γ envS → Corr ρ envS envT →
    ctx.indexBits = ib → CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
    RefinesC ResultsRefine (evalOps opSem ctx callT envT ts) (evalOps opSem ctx callS envS ss)

namespace TypedComparison

/-- The existing universal script checker remains the first comparison alternative. -/
def script : TypedComparison where
  run _ _ ρ ss ts := runScript ρ (synthLocal ρ ss ts) ss ts
  sound _ _ ρ ss ts ctx callS callT envS envT hr _ hc _ hcall :=
    OutLE.refinesC (runScript_sound callS callT ctx ρ _ ss ts envS envT hr hc hcall)

/-- The same universal script checker on scripts that erase dead source operations first: every
transitively dead one (dead-code pruning), or only the unused ones (what upstream `cse` erases). -/
def pruning : TypedComparison where
  run _ _ ρ ss ts := runScript ρ (synthLocalWith (deadIn ss) ρ ss ts) ss ts ||
    runScript ρ (synthLocalWith (unusedIn ss) ρ ss ts) ss ts
  sound _ _ ρ ss ts ctx callS callT envS envT hr _ hc _ hcall := by
    simp only [Bool.or_eq_true] at hr
    rcases hr with hr | hr
    · exact OutLE.refinesC (runScript_sound callS callT ctx ρ _ ss ts envS envT hr hc hcall)
    · exact OutLE.refinesC (runScript_sound callS callT ctx ρ _ ss ts envS envT hr hc hcall)

/-- Refusal of one comparison permits a separately proved alternative. Neither checker treats
the other's decision or an external certificate as a semantic premise. -/
def orElse (first second : TypedComparison) : TypedComparison where
  run ib Γ ρ ss ts := first.run ib Γ ρ ss ts || second.run ib Γ ρ ss ts
  sound ib Γ ρ ss ts ctx callS callT envS envT hr he hc hi hcall := by
    simp only [Bool.or_eq_true] at hr
    rcases hr with hr | hr
    · exact first.sound ib Γ ρ ss ts ctx callS callT envS envT hr he hc hi hcall
    · exact second.sound ib Γ ρ ss ts ctx callS callT envS envT hr he hc hi hcall

end TypedComparison

end MerlinVerify.Check
