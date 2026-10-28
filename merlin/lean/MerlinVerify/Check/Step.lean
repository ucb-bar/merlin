/-
One-op simulation step, used to interleave alpha-equivalent ops with rewrite sites (C1).
-/
import MerlinVerify.Check.Alpha

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- Execute one non-terminator op: read operands, run it, bind its results. -/
def stepOp (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op) : Outcome Env :=
  (env.getAll op.operands).bind fun args =>
  (sem ctx call op args (evalRegions sem call env op.regions)).bind fun results =>
  env.bindAll op.results results

theorem Outcome.bind_assoc {α β γ : Type} (x : Outcome α) (f : α → Outcome β) (g : β → Outcome γ) :
    (x.bind f).bind g = x.bind fun a => (f a).bind g := by
  cases x <;> rfl

theorem evalOps_cons_step (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op)
    (rest : List Op) (h : op.isTerminator = false) :
    evalOps sem ctx call env (op :: rest) =
      (stepOp sem ctx call env op).bind fun env' => evalOps sem ctx call env' rest := by
  rw [evalOps_cons_nonterm sem ctx call env op rest h, stepOp, Outcome.bind_assoc]
  congr 1; funext args
  rw [Outcome.bind_assoc]

theorem execOps_cons_step (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op)
    (rest : List Op) (h : op.isTerminator = false) :
    execOps sem ctx call env (op :: rest) =
      (stepOp sem ctx call env op).bind fun env' => execOps sem ctx call env' rest := by
  simp only [execOps, h, Bool.false_eq_true, ↓reduceIte, stepOp, Outcome.bind_assoc]

/-- Two step outcomes agree: both UB, both stuck, or both defined with corresponding envs. -/
def StepRel (ρ' : Rho) (t s : Outcome Env) : Prop :=
  (s = .ub ∧ t = .ub) ∨ ((∃ r, s = .stuck r) ∧ (∃ r, t = .stuck r)) ∨
    (∃ eS eT, s = .ok eS ∧ t = .ok eT ∧ Corr ρ' eS eT)

theorem alphaOp_spec {ρ ρ' : Rho} {s t : Op} (h : alphaOp ρ s t = some ρ') :
    opHeadEq s t = true ∧ s.operands.mapM ρ.find = some t.operands ∧
      alphaRegions ρ s.regions t.regions = true ∧ freshIds ρ (t.results.map (·.id)) = true ∧
      ρ' = (s.results.map (·.id)).zip (t.results.map (·.id)) ++ ρ := by
  cases s with
  | mk nS resS opsS aS pS regsS uS =>
    cases t with
    | mk nT resT opsT aT pT regsT uT =>
      simp only [alphaOp] at h
      split at h
      · next hc =>
        simp only [Option.some.injEq] at h
        simp only [Bool.and_eq_true, beq_iff_eq] at hc
        exact ⟨hc.1.1.1, hc.1.1.2, hc.1.2, hc.2, h.symm⟩
      · simp at h

/-- Alpha-equivalent non-terminator ops step in lockstep. -/
theorem alphaOp_step (callS callT : CallSem) {ρ ρ' : Rho} {s t : Op} (ctx : Ctx) {envS envT : Env}
    (h : alphaOp ρ s t = some ρ') (hc : Corr ρ envS envT)
    (hcallOp : ∀ c, calleeOf s.blind = some c → callS c = callT c)
    (hcallR : CallAgree callS callT (calleesRegions s.regions)) :
    StepRel ρ' (stepOp opSem ctx callT envT t) (stepOp opSem ctx callS envS s) := by
  obtain ⟨hhead, hmap, hregs, hfresh, rfl⟩ := alphaOp_spec h
  obtain ⟨hnd, hfr⟩ := freshIds_sound hfresh
  have hblind := blind_of_opHeadEq hhead
  have hlen := results_length_of_opHeadEq hhead
  unfold stepOp
  rw [Corr.getAll hc _ _ hmap, alphaRegions_sound callS callT ρ _ _ envS envT hregs hc hcallR]
  cases envT.getAll t.operands with
  | ub => exact Or.inl ⟨rfl, rfl⟩
  | stuck r => exact Or.inr (Or.inl ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩)
  | ok args =>
    simp only [Outcome.bind]
    rw [opSem_call_local (call' := callT) hcallOp, opSem_blind (op' := t) hblind]
    cases opSem ctx callT t args (evalRegions opSem callT envT t.regions) with
    | ub => exact Or.inl ⟨rfl, rfl⟩
    | stuck r => exact Or.inr (Or.inl ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩)
    | ok results =>
      simp only
      by_cases hk : s.results.length = results.length
      · rw [bindAll_eq_ok hk, bindAll_eq_ok (hlen ▸ hk)]
        exact Or.inr (Or.inr ⟨_, _, rfl, rfl,
          Corr.extend hc _ _ results (by simpa using hk) (by simpa [← hlen] using hk) hnd hfr⟩)
      · rw [bindAll_eq_stuck hk, bindAll_eq_stuck (fun e => hk (hlen ▸ e))]
        exact Or.inr (Or.inl ⟨⟨_, rfl⟩, ⟨_, rfl⟩⟩)

end MerlinVerify.Check
