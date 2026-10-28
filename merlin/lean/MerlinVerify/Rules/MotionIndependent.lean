/-
Adjacent independent pure expressions may exchange order. Unlike closed Motion, operands may
refer to prior bindings. Source-success decomposition and symmetric actual SSA dependency guards
supply both reordered successful steps. Source UB/stuck remain explicit in the universal SiteRel;
no totality, typing, floating algebra or semantic assumptions are added.
-/
import MerlinVerify.Rules.Motion

namespace MerlinVerify.Rules.MotionIndependent

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- An unrelated new binding changes none of the actual operand lookups, including failures. -/
theorem getAll_cons_not_mem (env : Env) (x : ValueId) (v : RVal) :
    ∀ ids, x ∉ ids → Env.getAll ((x, v) :: env) ids = Env.getAll env ids
  | [], _ => rfl
  | y :: ys, hn => by
    have hne : y ≠ x := fun he => hn (he ▸ List.mem_cons_self ..)
    have hb : (y == x) = false := by simpa using hne
    have ht := getAll_cons_not_mem env x v ys (fun hm => hn (List.mem_cons_of_mem _ hm))
    simp only [Env.getAll] at ht ⊢
    simp only [Outcome.mapM, Env.get?, List.lookup, hb]
    simp only [Env.get?, List.lookup] at ht
    rw [ht]

/-- Actual successful evaluation of a region-free single-result op supplies its operand values
and result. No fact is inferred from a certificate or an output type annotation. -/
theorem single_success {sem : OpSem} {ctx : Ctx} {call : CallSem} {op : Op}
    {env out : Env} {r : Value} (hr : op.regions = []) (hid : op.results = [r])
    (h : stepOp sem ctx call env op = .ok out) :
    ∃ args v, env.getAll op.operands = .ok args ∧ sem ctx call op args [] = .ok [v] ∧
      out = (r.id, v) :: env := by
  unfold stepOp at h
  rw [hr] at h
  simp only [evalRegions] at h
  cases hg : env.getAll op.operands with
  | ub => simp [hg] at h
  | stuck reason => simp [hg] at h
  | ok args =>
    rw [hg, Outcome.bind_ok] at h
    cases hs : sem ctx call op args [] with
    | ub => simp [hs] at h
    | stuck reason => simp [hs] at h
    | ok vs =>
      cases vs with
      | nil => simp [hs, Env.bindAll, hid] at h
      | cons v vs =>
        cases vs with
        | nil => exact ⟨args, v, rfl, hs, by simpa [hs, Env.bindAll, hid] using h.symm⟩
        | cons w ws => simp [hs, Env.bindAll, hid] at h

/-- Reconstruct a successful step from independently established actual operands and result. -/
theorem single_step {sem : OpSem} {ctx : Ctx} {call : CallSem} {op : Op}
    {env : Env} {r : Value} {args : List RVal} {v : RVal}
    (hr : op.regions = []) (hid : op.results = [r])
    (hg : env.getAll op.operands = .ok args) (hs : sem ctx call op args [] = .ok [v]) :
    stepOp sem ctx call env op = .ok ((r.id, v) :: env) := by
  simp [stepOp, hr, evalRegions, hg, hs, Env.bindAll, hid]

/-- The checker supplies Rule.sound's universal SiteRel: source UB/stuck or corresponding
successful executions of the two actual programs. Independent SSA guards are checked both ways. -/
def check (ρ : Rho) : List Op → List Op → Option Rho
  | [a, b], [t, u] =>
    match a.results, b.results with
    | [ra], [rb] =>
      if a.isTerminator = false ∧ b.isTerminator = false ∧
          Dce.pureOp a = true ∧ Dce.pureOp b = true ∧
          a.regions = [] ∧ b.regions = [] ∧
          calleeOf a.blind = none ∧ calleeOf b.blind = none ∧ ra.id ≠ rb.id ∧
          ra.id ∉ b.operands ∧ rb.id ∉ a.operands ∧
          ra.id ∉ a.operands ∧ rb.id ∉ b.operands then do
        let ρ₁ ← alphaOp ρ b t
        alphaOp ρ₁ a u
      else none
    | _, _ => none
  | _, _ => none

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [a, b], [t, u], ρ', h => by
    simp only [check] at h
    split at h
    · rename_i ra rb ha hb
      split at h
      · next hc =>
        obtain ⟨hat, hbt, _, _, har, hbr, hac, hbc, hne, hab, hba, _, _⟩ := hc
        cases hαb : alphaOp ρ b t with
        | none => simp [hαb] at h
        | some ρ₁ =>
          simp only [hαb] at h
          have htt : t.isTerminator = false :=
            (isTerminator_of_opHeadEq (alphaOp_spec hαb).1) ▸ hbt
          have hut : u.isTerminator = false :=
            (isTerminator_of_opHeadEq (alphaOp_spec h).1) ▸ hat
          have hnS : ∀ op ∈ [a, b], op.isTerminator = false := by
            simp only [List.mem_cons, List.not_mem_nil, or_false]
            intro op hm; rcases hm with rfl | rfl <;> assumption
          have hnT : ∀ op ∈ [t, u], op.isTerminator = false := by
            simp only [List.mem_cons, List.not_mem_nil, or_false]
            intro op hm; rcases hm with rfl | rfl <;> assumption
          refine ⟨hnS, hnT, fun ctx callS callT envS envT hcorr _ => ?_⟩
          rw [execOps_cons_step _ _ _ _ _ _ hat]
          cases hsa : stepOp opSem ctx callS envS a with
          | ub => exact Or.inl rfl
          | stuck reason => exact Or.inr (Or.inl ⟨reason, rfl⟩)
          | ok envA =>
            simp only [Outcome.bind_ok]
            rw [execOps_cons_step _ _ _ _ _ _ hbt]
            cases hsb : stepOp opSem ctx callS envA b with
            | ub => exact Or.inl rfl
            | stuck reason => exact Or.inr (Or.inl ⟨reason, rfl⟩)
            | ok envAB =>
              obtain ⟨argsA, va, hgA, hsA, rfl⟩ := single_success har ha hsa
              obtain ⟨argsB, vb, hgB, hsB, rfl⟩ := single_success hbr hb hsb
              have hgB' : envS.getAll b.operands = .ok argsB := by
                rw [getAll_cons_not_mem envS ra.id va b.operands hab] at hgB
                exact hgB
              have hbs := single_step hbr hb hgB' hsB
              obtain ⟨envT₁, hts, hc₁⟩ := Motion.alpha_success hαb hcorr hbr hbc hbs
              have hgA' : Env.getAll ((rb.id, vb) :: envS) a.operands = .ok argsA := by
                rw [getAll_cons_not_mem envS rb.id vb a.operands hba]
                exact hgA
              have has := single_step har ha hgA' hsA
              obtain ⟨envT₂, htu, hc₂⟩ := Motion.alpha_success h hc₁ har hac has
              refine Or.inr (Or.inr ⟨_, envT₂, rfl, ?_, fun x y hxy => ?_⟩)
              · rw [execOps_cons_step _ _ _ _ _ _ htt, hts, Outcome.bind_ok,
                  execOps_cons_step _ _ _ _ _ _ hut, htu]; rfl
              · exact (Motion.lookup_swap envS ra.id rb.id va vb hne x).symm.trans (hc₂ x y hxy)
      · cases h
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, [_], _, _, h => by simp [check] at h
  | _, _ :: _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_, _], [], _, h => by simp [check] at h
  | _, [_, _], [_], _, h => by simp [check] at h
  | _, [_, _], _ :: _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := { name := "motion-independent", check, sound := check_sound }

end MerlinVerify.Rules.MotionIndependent
