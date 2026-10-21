/- Composing checked actual straight-line segments through the frozen evaluator. -/
import MerlinVerify.Check.ProgramLemmas
import MerlinVerify.Rules.FrontendBoolSimulationWith

namespace MerlinVerify.Rules.FrontendProgramSegments
open MerlinVerify MerlinVerify.Check

def returned (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []

theorem returned_eval (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (ids : List Nat) :
    evalOps sem ctx call env [returned ids] = env.getAll ids := by
  simp [evalOps, returned, Op.isTerminator, terminatorNames, Op.name]

/-- Successful projected execution produces an actual intermediate SSA environment, with the
same evaluator and operation meanings as the surrounding program. -/
theorem prefix_state {sem : OpSem} {ctx : Ctx} {call : CallSem} {env : Env}
    {pre : List Op} {ids : List Nat} {values : List RVal}
    (hn : ∀ op ∈ pre, op.isTerminator = false)
    (h : evalOps sem ctx call env (pre ++ [returned ids]) = .ok values) :
    ∃ next, execOps sem ctx call env pre = .ok next ∧ next.getAll ids = .ok values := by
  rw [evalOps_append sem ctx call env pre [returned ids] hn] at h
  cases hx : execOps sem ctx call env pre with
  | ub => simp [hx, Outcome.bind] at h
  | stuck why => simp [hx, Outcome.bind] at h
  | ok next =>
    simp only [hx, Outcome.bind_ok] at h
    exact ⟨next, rfl, (returned_eval sem ctx call next ids) ▸ h⟩

theorem roots_corr (env : Env) (a b c : Nat) (mask values seed : RVal)
    (h : env.getAll [a,b,c] = .ok [mask,values,seed]) :
    Corr [(50,a),(51,b),(52,c)] [(50,mask),(51,values),(52,seed)] env := by
  cases ha : env.get? a <;> cases hb : env.get? b <;> cases hc : env.get? c <;>
    simp [Env.getAll, Outcome.mapM, Outcome.ofOption, Outcome.bind, ha, hb, hc] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  intro x y hx
  by_cases h0 : x = 50
  · subst x; simp [Rho.find, List.lookup] at hx; subst y; exact ha.symm
  · by_cases h1 : x = 51
    · subst x; simp [Rho.find, List.lookup] at hx; subst y; exact hb.symm
    · by_cases h2 : x = 52
      · subst x; simp [Rho.find, List.lookup] at hx; subst y; exact hc.symm
      · simp only [Rho.find, List.lookup, beq_eq_false_iff_ne.mpr h0,
          beq_eq_false_iff_ne.mpr h1, beq_eq_false_iff_ne.mpr h2] at hx
        cases hx

#print axioms prefix_state
#print axioms roots_corr
end MerlinVerify.Rules.FrontendProgramSegments
