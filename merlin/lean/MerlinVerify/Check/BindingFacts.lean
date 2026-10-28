/- Runtime facts with explicit SSA dependencies. A successful operation invalidates every fact
that depends on a rebound identifier; facts never acquire meaning from result types alone. -/
import MerlinVerify.Check.Step

namespace MerlinVerify.Check.BindingFacts

open MerlinVerify

/-- An extensible fact domain must state every binding its interpretation depends on. -/
structure Model (Fact : Type) where
  dependencies : Fact → List ValueId
  holds : Fact → Env → Prop
  congr : ∀ f env env', (∀ x ∈ dependencies f, env.get? x = env'.get? x) →
    holds f env → holds f env'

def Valid {Fact : Type} (M : Model Fact) (facts : List Fact) (env : Env) : Prop :=
  ∀ f ∈ facts, M.holds f env

def forget {Fact : Type} (M : Model Fact) (written : List ValueId) (facts : List Fact) : List Fact :=
  facts.filter fun f => (M.dependencies f).all fun x => !written.contains x

theorem lookup_zip_append {x : ValueId} {env : Env} :
    ∀ (ids : List ValueId) (values : List RVal), x ∉ ids →
      Env.get? (ids.zip values ++ env) x = env.get? x
  | [], _, _ => rfl
  | _ :: _, [], _ => rfl
  | y :: ys, v :: vs, h => by
    have hxy : x ≠ y := fun he => h (by simp [he])
    have hxy' : (x == y) = false := by simpa using hxy
    have ht : x ∉ ys := fun hm => h (List.mem_cons_of_mem _ hm)
    simpa [Env.get?, List.lookup, hxy'] using lookup_zip_append (env := env) ys vs ht

/-- Retained facts survive arbitrary result values, including poison and duplicate writes. -/
theorem forget_sound {Fact : Type} (M : Model Fact) {facts : List Fact} {env : Env}
    (h : Valid M facts env) (ids : List ValueId) (values : List RVal) :
    Valid M (forget M ids facts) (ids.zip values ++ env) := by
  intro f hf
  have hf' : f ∈ facts ∧ ∀ x ∈ M.dependencies f, x ∉ ids := by
    simpa [forget, List.all_eq_true] using hf
  apply M.congr f env _ _ (h f hf'.1)
  intro x hx
  exact (lookup_zip_append ids values (hf'.2 x hx)).symm

/-- Successful SSA execution only prepends its actual result bindings. -/
theorem step_bindings {sem : OpSem} {ctx : Ctx} {call : CallSem} {env env' : Env} {op : Op}
    (h : stepOp sem ctx call env op = .ok env') :
    ∃ values, env' = (op.results.map (·.id)).zip values ++ env := by
  unfold stepOp at h
  cases ha : env.getAll op.operands with
  | ub => simp [ha, Outcome.bind] at h
  | stuck r => simp [ha, Outcome.bind] at h
  | ok args =>
    rw [ha, Outcome.bind_ok] at h
    cases hv : sem ctx call op args (evalRegions sem call env op.regions) with
    | ub => simp [hv, Outcome.bind] at h
    | stuck r => simp [hv, Outcome.bind] at h
    | ok values =>
      rw [hv, Outcome.bind_ok] at h
      unfold Env.bindAll at h
      split at h
      · exact ⟨values, (Outcome.ok.inj h).symm⟩
      · cases h

/-- No stale fact survives a successful operation that shadows one of its dependencies. -/
theorem forget_step {Fact : Type} (M : Model Fact) {facts : List Fact}
    {sem : OpSem} {ctx : Ctx} {call : CallSem} {env env' : Env} {op : Op}
    (hf : Valid M facts env) (hs : stepOp sem ctx call env op = .ok env') :
    Valid M (forget M (op.results.map (·.id)) facts) env' := by
  obtain ⟨values, rfl⟩ := step_bindings hs
  exact forget_sound M hf _ values

end MerlinVerify.Check.BindingFacts
