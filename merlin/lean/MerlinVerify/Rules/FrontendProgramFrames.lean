/- Intermediate SSA facts and unchanged captures across an actual operation segment. -/
import MerlinVerify.Rules.FrontendRangeFacts
import MerlinVerify.Rules.Dce

namespace MerlinVerify.Rules.FrontendProgramFrames
open MerlinVerify MerlinVerify.Check
open FrontendTensorRanges

theorem getAll_cons (env : Env) (x : ValueId) (xs : List ValueId) :
    env.getAll (x :: xs) = (Outcome.ofOption "unbound SSA value" (env.get? x)).bind fun v =>
      (env.getAll xs).bind fun vs => .ok (v :: vs) := rfl

/-- Facts learned from actual returned SSA values hold in that same intermediate environment. -/
theorem of_getAll {ib : Nat} (env : Env) : ∀ ids vs fs,
    env.getAll ids = .ok vs → Definedness.Refined.FactsHold model ib vs fs →
    Definedness.Refined.EnvHolds model ib (ids.zip fs) env
  | [], vs, fs, h, hf => by
    cases h
    cases hf
    intro _ _ h; cases h
  | x :: xs, vs, fs, h, hf => by
    rw [getAll_cons] at h
    cases hx : env.get? x with
    | none => simp [hx, Outcome.ofOption, Outcome.bind] at h
    | some v =>
      simp only [hx, Outcome.ofOption, Outcome.bind_ok] at h
      cases hxs : env.getAll xs with
      | ub => rw [hxs] at h; cases h
      | stuck why => rw [hxs] at h; cases h
      | ok vals =>
        rw [hxs] at h
        cases h
        obtain _ | ⟨hv, hrest⟩ := hf
        have ih := of_getAll env xs vals _ hxs hrest
        intro y f hy
        simp only [List.zip_cons_cons, List.lookup] at hy
        split at hy
        · next heq =>
          have heq' : y = x := beq_iff_eq.mp heq
          subst y
          cases hy
          exact ⟨v, hx, hv⟩
        · exact ih y f hy

def preserves (Γ : FEnv) (op : Op) : Bool := op.results.all fun v => (Γ.lookup v.id).isNone

theorem step_preserves {ib : Nat} {Γ : FEnv} {sem : OpSem} {ctx : Ctx} {call : CallSem}
    {env next : Env} {op : Op} (hΓ : Definedness.Refined.EnvHolds model ib Γ env)
    (hp : preserves Γ op = true) (he : stepOp sem ctx call env op = .ok next) :
    Definedness.Refined.EnvHolds model ib Γ next := by
  obtain ⟨vs, rfl⟩ := Dce.stepOp_ok_form he
  intro x f hx
  obtain ⟨v, hv, hf⟩ := hΓ x f hx
  refine ⟨v, ?_, hf⟩
  have hn : x ∉ op.results.map (·.id) := by
    intro hmem
    obtain ⟨r, hr, heq⟩ := List.mem_map.mp hmem
    have hnone := List.all_eq_true.mp hp r hr
    rw [heq, hx] at hnone
    cases hnone
  simpa only [Env.get?, Dce.lookup_zip_append_not_mem x _ vs env hn] using hv

theorem exec_preserves {ib : Nat} {Γ : FEnv} {sem : OpSem} {ctx : Ctx} {call : CallSem} :
    ∀ ops env next, Definedness.Refined.EnvHolds model ib Γ env →
      (∀ op ∈ ops, op.isTerminator = false) → ops.all (preserves Γ) = true →
      execOps sem ctx call env ops = .ok next → Definedness.Refined.EnvHolds model ib Γ next
  | [], env, next, hΓ, _, _, he => by cases he; exact hΓ
  | op :: ops, env, next, hΓ, hn, hp, he => by
    rw [execOps_cons_step sem ctx call env op ops (hn op (List.mem_cons_self ..))] at he
    cases hs : stepOp sem ctx call env op with
    | ub => simp [hs, Outcome.bind] at he
    | stuck why => simp [hs, Outcome.bind] at he
    | ok mid =>
      simp only [List.all_cons, Bool.and_eq_true] at hp
      simp only [hs, Outcome.bind_ok] at he
      exact exec_preserves ops mid next (step_preserves hΓ hp.1 hs)
        (fun op hmem => hn op (List.mem_cons_of_mem _ hmem)) hp.2 he

/-- Taking a prefix of an actual multi-value observation preserves the read proof. -/
theorem getAll_take (env : Env) : ∀ ids vs, env.getAll ids = .ok vs → ∀ n,
    env.getAll (ids.take n) = .ok (vs.take n)
  | ids, vs, h, 0 => rfl
  | [], vs, h, _ + 1 => by cases h; rfl
  | x :: xs, vs, h, n + 1 => by
    rw [getAll_cons] at h
    cases hx : env.get? x with
    | none => simp [hx, Outcome.ofOption, Outcome.bind] at h
    | some v =>
      simp only [hx, Outcome.ofOption, Outcome.bind_ok] at h
      cases hxs : env.getAll xs with
      | ub => rw [hxs] at h; cases h
      | stuck why => rw [hxs] at h; cases h
      | ok vals =>
        rw [hxs] at h
        cases h
        have ih := getAll_take env xs vals hxs n
        simp only [List.take_succ_cons, getAll_cons, hx, Outcome.ofOption, Outcome.bind_ok, ih]

theorem append {ib : Nat} {Γ Δ : FEnv} {env : Env}
    (hΓ : Definedness.Refined.EnvHolds model ib Γ env) (hΔ : Definedness.Refined.EnvHolds model ib Δ env) :
    Definedness.Refined.EnvHolds model ib (Γ ++ Δ) env := by
  intro x f hx
  rw [List.lookup_append] at hx
  cases hh : Γ.lookup x with
  | some g => simp only [hh, Option.or, Option.some.injEq] at hx; subst f; exact hΓ x g hh
  | none => simp only [hh, Option.or] at hx; exact hΔ x f hx

theorem region_eq (limit : Nat) (ctx : Ctx) (call : CallSem) (env : Env) (r : Region)
    (args : List RVal) (h : FrontendRuntimeBridge.blocksCheck r.blocks = true) :
    evalRegion (FrontendMaskRuntime.sem limit) ctx call env r args =
      evalRegion Sem.Ops.opSem ctx call env r args := by
  cases r with
  | mk blocks => exact FrontendRuntimeBridge.blocks_sound limit ctx call env blocks args h

#print axioms exec_preserves
end MerlinVerify.Rules.FrontendProgramFrames
