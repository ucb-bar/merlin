/-
Swap adjacent closed, pure expressions. Each operation has one result, no operands and no
regions; distinct source result ids make their bindings commute. Alpha checking in target
execution order checks both operations and fresh target bindings. No arithmetic is evaluated.
-/
import MerlinVerify.Rules.Cse

namespace MerlinVerify.Rules.Motion

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def check (ρ : Rho) : List Op → List Op → Option Rho
  | [a, b], [t, u] =>
    match a.results, b.results with
    | [ra], [rb] =>
      if a.isTerminator = false ∧ b.isTerminator = false ∧
          Dce.pureOp a = true ∧ Dce.pureOp b = true ∧
          a.operands = [] ∧ b.operands = [] ∧ a.regions = [] ∧ b.regions = [] ∧
          calleeOf a.blind = none ∧ calleeOf b.blind = none ∧ ra.id ≠ rb.id then do
        let ρ₁ ← alphaOp ρ b t
        alphaOp ρ₁ a u
      else none
    | _, _ => none
  | _, _ => none

/-- Successful execution of a closed single-result expression reveals its environment-independent
value, including the checked result arity. -/
theorem closed_success {ctx : Ctx} {call : CallSem} {op : Op} {env out : Env} {r : Value}
    (ho : op.operands = []) (hr : op.regions = []) (hid : op.results = [r])
    (h : stepOp opSem ctx call env op = .ok out) :
    ∃ v, opSem ctx call op [] [] = .ok [v] ∧ out = (r.id, v) :: env := by
  rw [Cse.closedStep ho hr] at h
  cases hs : opSem ctx call op [] [] with
  | ub => simp [hs] at h
  | stuck reason => simp [hs] at h
  | ok vs =>
    cases vs with
    | nil => simp [hs, Env.bindAll, hid] at h
    | cons v vs =>
      cases vs with
      | nil => exact ⟨v, rfl, by simpa [hs, Env.bindAll, hid] using h.symm⟩
      | cons w ws => simp [hs, Env.bindAll, hid] at h

/-- Specialize the existing alpha simulation to a successful, region-free non-call. -/
theorem alpha_success {ctx : Ctx} {callS callT : CallSem} {ρ ρ' : Rho}
    {s t : Op} {envS envT out : Env}
    (ha : alphaOp ρ s t = some ρ') (hc : Corr ρ envS envT)
    (hr : s.regions = []) (hn : calleeOf s.blind = none)
    (hs : stepOp opSem ctx callS envS s = .ok out) :
    ∃ outT, stepOp opSem ctx callT envT t = .ok outT ∧ Corr ρ' out outT := by
  have h := alphaOp_step callS callT ctx ha hc
    (by intro c he; rw [hn] at he; cases he)
    (by simp [hr, CallAgree, calleesRegions])
  rcases h with ⟨h, _⟩ | ⟨⟨reason, h⟩, _⟩ | ⟨eS, eT, hS, hT, hcorr⟩
  · rw [hs] at h; cases h
  · rw [hs] at h; cases h
  · rw [hs] at hS
    cases hS
    exact ⟨eT, hT, hcorr⟩

theorem lookup_swap (env : Env) (a b : ValueId) (va vb : RVal) (hne : a ≠ b) (x : ValueId) :
    Env.get? ((a, va) :: (b, vb) :: env) x =
      Env.get? ((b, vb) :: (a, va) :: env) x := by
  by_cases ha : x = a
  · subst ha
    have he : (x == b) = false := by simpa using hne
    simp [Env.get?, List.lookup, he]
  · by_cases hb : x = b
    · subst hb
      have he : (x == a) = false := by simpa using Ne.symm hne
      simp [Env.get?, List.lookup, he]
    · have hea : (x == a) = false := by simpa using ha
      have heb : (x == b) = false := by simpa using hb
      simp [Env.get?, List.lookup, hea, heb]

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
        obtain ⟨hat, hbt, _, _, hao, hbo, har, hbr, hac, hbc, hne⟩ := hc
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
              obtain ⟨va, hva, rfl⟩ := closed_success hao har ha hsa
              obtain ⟨vb, hvb, rfl⟩ := closed_success hbo hbr hb hsb
              have hbs : stepOp opSem ctx callS envS b = .ok ((rb.id, vb) :: envS) := by
                rw [Cse.closedStep hbo hbr, hvb]; simp [Env.bindAll, hb]
              obtain ⟨envT₁, hts, hc₁⟩ := alpha_success hαb hcorr hbr hbc hbs
              have has : stepOp opSem ctx callS ((rb.id, vb) :: envS) a =
                  .ok ((ra.id, va) :: (rb.id, vb) :: envS) := by
                rw [Cse.closedStep hao har, hva]; simp [Env.bindAll, ha]
              obtain ⟨envT₂, htu, hc₂⟩ := alpha_success h hc₁ har hac has
              refine Or.inr (Or.inr ⟨_, envT₂, rfl, ?_, fun x y hxy => ?_⟩)
              · rw [execOps_cons_step _ _ _ _ _ _ htt, hts, Outcome.bind_ok,
                  execOps_cons_step _ _ _ _ _ _ hut, htu]; rfl
              · exact (lookup_swap envS ra.id rb.id va vb hne x).symm.trans (hc₂ x y hxy)
      · cases h
    · cases h
  | _, [], _, _, h => by simp [check] at h
  | _, [_], _, _, h => by simp [check] at h
  | _, _ :: _ :: _ :: _, _, _, h => by simp [check] at h
  | _, [_, _], [], _, h => by simp [check] at h
  | _, [_, _], [_], _, h => by simp [check] at h
  | _, [_, _], _ :: _ :: _ :: _, _, h => by simp [check] at h

def rule : Rule := { name := "motion-closed", check, sound := check_sound }

end MerlinVerify.Rules.Motion
