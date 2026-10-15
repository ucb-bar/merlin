/- Lift a proved operation expansion through SSA renaming and the existing C1 rule interface. -/
import MerlinVerify.Check.C1

namespace MerlinVerify.Rules

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- An expansion preserves SSA bindings and region closures. Its meaning must agree on every
modeled, defined source execution, including poison values and arbitrary floating-point models. -/
structure Expansion where
  expand : Op → Option Op
  sound : ∀ s v, expand s = some v →
    s.isTerminator = false ∧ v.isTerminator = false ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧ calleeOf v.blind = none ∧
    ∀ ctx call args regions, OutLE (opSem ctx call v args regions) (opSem ctx call s args regions)

namespace Expansion

theorem step_agrees {s v : Op} (ho : v.operands = s.operands) (hr : v.regions = s.regions)
    (hv : v.results = s.results)
    (hsem : ∀ ctx call args regions,
      OutLE (opSem ctx call v args regions) (opSem ctx call s args regions))
    (ctx : Ctx) (call : CallSem) (env : Env) :
    stepOp opSem ctx call env s = .ub ∨
    (∃ r, stepOp opSem ctx call env s = .stuck r) ∨
    stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  unfold stepOp
  rw [ho, hr, hv]
  cases hg : env.getAll s.operands with
  | ub => exact Or.inl rfl
  | stuck r => exact Or.inr (Or.inl ⟨r, rfl⟩)
  | ok args =>
    simp only [Outcome.bind_ok]
    rcases hsem ctx call args (evalRegions opSem call env s.regions) with hu | ⟨r, hs⟩ | he
    · rw [hu]; exact Or.inl rfl
    · rw [hs]; exact Or.inr (Or.inl ⟨r, rfl⟩)
    · rw [he]; exact Or.inr (Or.inr rfl)

def check (E : Expansion) (ρ : Rho) : List Op → List Op → Option Rho
  | [s], [t] => do
    let v ← E.expand s
    alphaOp ρ v t
  | _, _ => none

theorem check_sound (E : Expansion) : ∀ ρ ss ts ρ', check E ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [s], [t], ρ', h => by
    simp only [check] at h
    cases he : E.expand s with
    | none => simp [he] at h
    | some v =>
      simp only [he] at h
      obtain ⟨hst, hvt, ho, hv, hr, hcallee, hsem⟩ := E.sound s v he
      have htt : t.isTerminator = false := (isTerminator_of_opHeadEq (alphaOp_spec h).1) ▸ hvt
      refine ⟨by simpa using hst, by simpa using htt,
        fun ctx callS callT envS envT hc hcall => ?_⟩
      have hcallR : CallAgree callS callT (calleesRegions v.regions) := by
        rw [hr]
        intro c hm
        apply hcall c
        exact List.mem_append_left _ (by
          rw [calleesOps_cons]
          exact List.mem_append_left _ (List.mem_append_right _ hm))
      rw [execOps_cons_step _ _ _ _ _ _ hst, execOps_cons_step _ _ _ _ _ _ htt]
      rcases step_agrees ho hr hv hsem ctx callS envS with hu | ⟨r, hs⟩ | heq
      · rw [hu]; exact Or.inl rfl
      · rw [hs]; exact Or.inr (Or.inl ⟨r, rfl⟩)
      · rw [← heq]
        have hcallOp : ∀ c, calleeOf v.blind = some c → callS c = callT c := by
          intro c h; rw [hcallee] at h; cases h
        rcases alphaOp_step callS callT ctx h hc hcallOp hcallR with
          ⟨hs, _⟩ | ⟨⟨r, hs⟩, _⟩ | ⟨eS, eT, hs, ht, hcorr⟩
        · rw [hs]; exact Or.inl rfl
        · rw [hs]; exact Or.inr (Or.inl ⟨r, rfl⟩)
        · rw [hs, ht]
          exact Or.inr (Or.inr ⟨eS, eT, rfl, rfl, hcorr⟩)
  | _, [], _, _, h => by simp [check] at h
  | _, _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_], [], _, h => by simp [check] at h
  | _, [_], _ :: _ :: _, _, h => by simp [check] at h

def rule (E : Expansion) (name : String) : Rule := ⟨name, check E, check_sound E⟩

end Expansion
end MerlinVerify.Rules
