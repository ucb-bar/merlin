/-
Adjacent closed-expression CSE. Two identical, effect-free, region-free operations with no
operands may share one result. The rule checks actual operations and extends the SSA correspondence
for both source results. It neither evaluates floats nor assumes algebraic identities.
-/
import MerlinVerify.Rules.Dce

namespace MerlinVerify.Rules.Cse

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def check (ρ : Rho) : List Op → List Op → Option Rho
  | [a, b], [t] =>
    match a.results, b.results, t.results with
    | [_], [rb], [rt] =>
      if a.isTerminator = false ∧ Dce.pureOp a = true ∧ opHeadEq a b = true ∧
          a.operands = [] ∧ b.operands = [] ∧ t.operands = [] ∧
          a.regions = [] ∧ b.regions = [] ∧ t.regions = [] ∧ calleeOf a.blind = none then
        (alphaOp ρ a t).map fun ρ' => (rb.id, rt.id) :: ρ'
      else none
    | _, _, _ => none
  | _, _ => none

theorem Corr.bindAlias {ρ : Rho} {s t : Env} {x y : ValueId} {v : RVal}
    (hc : Corr ρ s t) (hv : t.get? y = some v) : Corr ((x, y) :: ρ) ((x, v) :: s) t := by
  intro a b h
  simp only [Rho.find, List.lookup] at h
  simp only [Env.get?, List.lookup]
  by_cases he : a = x
  · subst he
    simp only [beq_self_eq_true, Option.some.injEq] at h ⊢
    subst h
    exact hv.symm
  · have hb : (a == x) = false := by simpa using he
    simp only [hb] at h ⊢
    exact hc a b h

theorem closedStep {sem : OpSem} {ctx : Ctx} {call : CallSem} {op : Op} {env : Env}
    (ho : op.operands = []) (hr : op.regions = []) :
    stepOp sem ctx call env op =
      (sem ctx call op [] []).bind fun results => env.bindAll op.results results := by
  simp only [stepOp, ho, hr, Env.getAll, Outcome.mapM, Outcome.bind_ok, evalRegions]

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [a, b], [t], ρ', h => by
    simp only [check] at h
    split at h
    · rename_i ra rb rt ha hb ht
      split at h
      · next hc =>
        obtain ⟨hat, _, hab, hao, hbo, hto, har, hbr, htr, hcallee⟩ := hc
        cases halpha : alphaOp ρ a t with
        | none => simp [halpha] at h
        | some ρ1 =>
          simp only [halpha, Option.map_some, Option.some.injEq] at h
          subst h
          obtain ⟨hhead, _, _, hfr, hρ⟩ := alphaOp_spec halpha
          have hbt : b.isTerminator = false := (isTerminator_of_opHeadEq hab) ▸ hat
          have htt : t.isTerminator = false := (isTerminator_of_opHeadEq hhead) ▸ hat
          refine ⟨by simp only [List.mem_cons, List.not_mem_nil, or_false]; intro op hm; rcases hm with rfl | rfl <;> assumption,
            by simpa using htt, fun ctx callS callT envS envT hcorr _ => ?_⟩
          have hblindB := blind_of_opHeadEq hab
          have hblindT := blind_of_opHeadEq hhead
          have hcall : ∀ c, calleeOf a.blind = some c → callS c = callT c := by
            intro c he; rw [hcallee] at he; cases he
          have hsemB : opSem ctx callS b [] [] = opSem ctx callS a [] [] := opSem_blind hblindB.symm
          have hsemT : opSem ctx callT t [] [] = opSem ctx callS a [] [] := by
            rw [← opSem_blind hblindT, ← opSem_call_local hcall]
          rw [execOps_cons_step _ _ _ _ _ _ hat, closedStep hao har]
          cases he : opSem ctx callS a [] [] with
          | ub => exact Or.inl rfl
          | stuck reason => exact Or.inr (Or.inl ⟨reason, rfl⟩)
          | ok vs =>
            cases vs with
            | nil => exact Or.inr (Or.inl ⟨"arity mismatch binding SSA values",
                by simp [Outcome.bind_ok, Env.bindAll, ha]⟩)
            | cons v vs =>
              cases vs with
              | cons w ws => exact Or.inr (Or.inl ⟨"arity mismatch binding SSA values",
                  by simp [Outcome.bind_ok, Env.bindAll, ha]⟩)
              | nil =>
                have hs : stepOp opSem ctx callS ((ra.id, v) :: envS) b =
                    .ok ((rb.id, v) :: (ra.id, v) :: envS) := by
                  rw [closedStep hbo hbr, hsemB, he]
                  simp [Env.bindAll, hb]
                have htarget : execOps opSem ctx callT envT [t] = .ok ((rt.id, v) :: envT) := by
                  rw [execOps_cons_step _ _ _ _ _ _ htt, closedStep hto htr, hsemT, he]
                  simp [Env.bindAll, ht, execOps, Outcome.bind_ok]
                simp only [Outcome.bind_ok]
                have hfirst : envS.bindAll a.results [v] = .ok ((ra.id, v) :: envS) := by
                  simp [Env.bindAll, ha]
                rw [hfirst, Outcome.bind_ok, execOps_cons_step _ _ _ _ _ _ hbt, hs]
                simp only [Outcome.bind_ok, execOps]
                obtain ⟨hnd, hfresh⟩ := freshIds_sound hfr
                have hc1 : Corr ρ1 ((ra.id, v) :: envS) ((rt.id, v) :: envT) := by
                  rw [hρ]
                  have hc := Corr.extend hcorr [ra.id] [rt.id] [v] rfl rfl
                    (by simp) (by simpa [ht] using hfresh)
                  simpa [ha, ht] using hc
                exact Or.inr (Or.inr ⟨_, _, rfl, htarget,
                  Corr.bindAlias hc1 (by simp [Env.get?, List.lookup])⟩)
      · cases h
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, [_], _, _, h => by simp [check] at h
  | _, _ :: _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_, _], [], _, h => by simp [check] at h
  | _, [_, _], _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := { name := "cse-closed", check, sound := check_sound }

end MerlinVerify.Rules.Cse
