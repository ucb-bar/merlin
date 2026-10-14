/-
C1 — the rule-instance checker (design §5.1).

A boundary is checked by a script over one function body: `same` steps (one op on each side,
alpha-equivalent) and `site` steps (a contiguous run of source ops replaced by a run of target ops,
checked by a `Rule`). The script comes from the pass's certificate and is UNTRUSTED: `runScript`
re-checks every step against the IR, and `runScript_sound` holds for any script.

Claim (`c1Check_sound`): the target function refines the source function wherever the source is
modelled (`FuncRefinesC`), for every float model, every external environment and every input.
-/
import MerlinVerify.Check.Step

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops

/-- What a rewrite site must establish: the source site is UB or stuck, or both sites are
defined and the environments correspond under the extended `ρ'`. -/
def SiteRel (ρ' : Rho) (t s : Outcome Env) : Prop :=
  s = .ub ∨ (∃ r, s = .stuck r) ∨ (∃ eS eT, s = .ok eS ∧ t = .ok eT ∧ Corr ρ' eS eT)

/-- A rewrite rule: a decidable instance check and its soundness proof. -/
structure Rule where
  name : String
  check : Rho → List Op → List Op → Option Rho
  sound : ∀ ρ siteS siteT ρ', check ρ siteS siteT = some ρ' →
    (∀ op ∈ siteS, op.isTerminator = false) ∧ (∀ op ∈ siteT, op.isTerminator = false) ∧
    ∀ (ctx : Ctx) (callS callT : CallSem) (envS envT : Env), Corr ρ envS envT →
      CallAgree callS callT (calleesOps siteS ++ calleesOps siteT) →
      SiteRel ρ' (execOps opSem ctx callT envT siteT) (execOps opSem ctx callS envS siteS)

inductive Step where
  | same
  | site (rule : Rule) (nS nT : Nat)

def runScript : Rho → List Step → List Op → List Op → Bool
  | _, [], [], [] => true
  | ρ, .same :: st, s :: ss, t :: ts =>
    match alphaOp ρ s t with
    | some ρ' =>
      if s.isTerminator then ss.isEmpty && ts.isEmpty && st.isEmpty
      else runScript ρ' st ss ts
    | none => false
  | ρ, .site R nS nT :: st, ss, ts =>
    if (ss.take nS).length = nS ∧ (ts.take nT).length = nT then
      match R.check ρ (ss.take nS) (ts.take nT) with
      | some ρ' => runScript ρ' st (ss.drop nS) (ts.drop nT)
      | none => false
    else false
  | _, _, _, _ => false

/-- `t` agrees with `s` wherever `s` is defined and modelled. -/
def OutLE (t s : Outcome (List RVal)) : Prop :=
  s = .ub ∨ (∃ r, s = .stuck r) ∨ t = s

theorem OutLE.refinesC {t s : Outcome (List RVal)} (h : OutLE t s) : RefinesC ResultsRefine t s := by
  rcases h with rfl | ⟨r, rfl⟩ | rfl
  · exact Or.inr trivial
  · exact Or.inl ⟨r, rfl⟩
  · cases t with
    | ub => exact Or.inr trivial
    | stuck r => exact Or.inl ⟨r, rfl⟩
    | ok v => exact Or.inr (ResultsRefine.refl v)

theorem calleesOps_cons (op : Op) (rest : List Op) :
    calleesOps (op :: rest) = (calleeOf op.blind).toList ++ calleesRegions op.regions ++ calleesOps rest := by
  cases op; rw [calleesOps.eq_2]; rfl

theorem calleesOps_append : ∀ (a b : List Op), calleesOps (a ++ b) = calleesOps a ++ calleesOps b
  | [], b => by simp [calleesOps]
  | op :: a, b => by
    rw [List.cons_append, calleesOps_cons, calleesOps_cons, calleesOps_append a b]
    simp [List.append_assoc]

theorem evalOps_cons_term (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (op : Op)
    (rest : List Op) (h : op.isTerminator = true) :
    evalOps sem ctx call env (op :: rest) =
      if rest.isEmpty then env.getAll op.operands else .stuck "ops after terminator" := by
  cases op with
  | mk name results operands attrs props regions succ =>
    rw [evalOps.eq_2, ite_eq_left h]
    rfl

theorem CallAgree.mono {callS callT : CallSem} {a b : List String} (h : CallAgree callS callT b)
    (sub : ∀ c ∈ a, c ∈ b) : CallAgree callS callT a := fun c hc => h c (sub c hc)

theorem runScript_sound (callS callT : CallSem) (ctx : Ctx) :
    ∀ (ρ : Rho) (st : List Step) (ss ts : List Op) (envS envT : Env),
      runScript ρ st ss ts = true → Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      OutLE (evalOps opSem ctx callT envT ts) (evalOps opSem ctx callS envS ss)
  | _, [], [], [], _, _, _, _, _ => Or.inr (Or.inl ⟨_, rfl⟩)
  | ρ, .same :: st, s :: ss, t :: ts, envS, envT, h, hc, hcall => by
    simp only [runScript] at h
    split at h
    · next ρ' hop =>
      obtain ⟨hhead, -, -, -, -⟩ := alphaOp_spec hop
      have hterm := isTerminator_of_opHeadEq hhead
      have hcS : ∀ c ∈ calleesOps (s :: ss), c ∈ calleesOps (s :: ss) ++ calleesOps (t :: ts) :=
        fun c hc' => List.mem_append_left _ hc'
      by_cases hs : s.isTerminator = true
      · rw [ite_eq_left hs] at h
        simp only [Bool.and_eq_true, List.isEmpty_iff] at h
        obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
        have ht : t.isTerminator = true := hterm ▸ hs
        rw [evalOps_cons_term _ _ _ _ _ _ hs, evalOps_cons_term _ _ _ _ _ _ ht]
        obtain ⟨-, hmap, -⟩ := alphaOp_spec hop
        simp only [List.isEmpty_nil, ↓reduceIte]
        rw [Corr.getAll hc _ _ hmap]
        exact Or.inr (Or.inr rfl)
      · rw [ite_eq_right hs] at h
        have hs' : s.isTerminator = false := by simpa using hs
        have ht' : t.isTerminator = false := hterm ▸ hs'
        rw [evalOps_cons_step _ _ _ _ _ _ hs', evalOps_cons_step _ _ _ _ _ _ ht']
        have hcallOp : ∀ c, calleeOf s.blind = some c → callS c = callT c := fun c hc' =>
          hcall c (hcS c (by rw [calleesOps_cons]; simp [hc']))
        have hcallR : CallAgree callS callT (calleesRegions s.regions) := fun c hc' =>
          hcall c (hcS c (by rw [calleesOps_cons]; exact List.mem_append_left _ (List.mem_append_right _ hc')))
        have hcallRest : CallAgree callS callT (calleesOps ss ++ calleesOps ts) := by
          refine CallAgree.mono hcall fun c hc' => ?_
          rcases List.mem_append.1 hc' with h1 | h1
          · exact List.mem_append_left _ (by rw [calleesOps_cons]; exact List.mem_append_right _ h1)
          · exact List.mem_append_right _ (by rw [calleesOps_cons]; exact List.mem_append_right _ h1)
        rcases alphaOp_step callS callT ctx hop hc hcallOp hcallR with
          ⟨hsu, htu⟩ | ⟨⟨r, hsr⟩, ⟨r', htr⟩⟩ | ⟨eS, eT, hse, hte, hce⟩
        · rw [hsu]; exact Or.inl rfl
        · rw [hsr]; exact Or.inr (Or.inl ⟨r, rfl⟩)
        · rw [hse, hte]
          exact runScript_sound callS callT ctx ρ' st ss ts eS eT h hce hcallRest
    · simp at h
  | ρ, .site R nS nT :: st, ss, ts, envS, envT, h, hc, hcall => by
    simp only [runScript] at h
    split at h
    · next hlen =>
      split at h
      · next ρ' hchk =>
        obtain ⟨htS, htT, hsite⟩ := R.sound ρ _ _ ρ' hchk
        rw [← List.take_append_drop nS ss, ← List.take_append_drop nT ts,
          evalOps_append _ _ _ _ _ _ htS, evalOps_append _ _ _ _ _ _ htT]
        have hcallAll : CallAgree callS callT
            (calleesOps (ss.take nS) ++ calleesOps (ss.drop nS) ++
              (calleesOps (ts.take nT) ++ calleesOps (ts.drop nT))) := by
          rw [← calleesOps_append, ← calleesOps_append, List.take_append_drop, List.take_append_drop]
          exact hcall
        have hcallSite : CallAgree callS callT (calleesOps (ss.take nS) ++ calleesOps (ts.take nT)) :=
          CallAgree.mono hcallAll fun c hc' => by
            rcases List.mem_append.1 hc' with h1 | h1
            · exact List.mem_append_left _ (List.mem_append_left _ h1)
            · exact List.mem_append_right _ (List.mem_append_left _ h1)
        have hcallRest : CallAgree callS callT (calleesOps (ss.drop nS) ++ calleesOps (ts.drop nT)) :=
          CallAgree.mono hcallAll fun c hc' => by
            rcases List.mem_append.1 hc' with h1 | h1
            · exact List.mem_append_left _ (List.mem_append_right _ h1)
            · exact List.mem_append_right _ (List.mem_append_right _ h1)
        rcases hsite ctx callS callT envS envT hc hcallSite with hu | ⟨r, hr⟩ | ⟨eS, eT, hse, hte, hce⟩
        · rw [hu]; exact Or.inl rfl
        · rw [hr]; exact Or.inr (Or.inl ⟨r, rfl⟩)
        · rw [hse, hte]
          exact runScript_sound callS callT ctx ρ' st _ _ eS eT h hce hcallRest
      · simp at h
    · simp at h
  | _, [], _ :: _, _, _, _, h, _, _ => by simp [runScript] at h
  | _, [], [], _ :: _, _, _, h, _, _ => by simp [runScript] at h
  | _, .same :: _, [], _, _, _, h, _, _ => by simp [runScript] at h
  | _, .same :: _, _ :: _, [], _, _, h, _, _ => by simp [runScript] at h

/-! ### Function level -/

/-- `c` is a declaration (no body) in module `m`. -/
def isDeclIn (m : Module) (c : String) : Bool :=
  match m.func? c with
  | some f => f.isDecl
  | none => false

structure C1Input where
  src : Module
  tgt : Module
  fn : String
  script : List Step

def c1Check (i : C1Input) : Bool :=
  match i.src.func? i.fn, i.tgt.func? i.fn with
  | some fS, some fT =>
    match fS.body, fT.body with
    | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
      !fS.isDecl && !fT.isDecl &&
      argsS.map (·.ty) == argsT.map (·.ty) && freshIds [] (argsT.map (·.id)) &&
      (calleesOps opsS ++ calleesOps opsT).all (fun c => isDeclIn i.src c && isDeclIn i.tgt c) &&
      runScript ((argsS.map (·.id)).zip (argsT.map (·.id))) i.script opsS opsT
    | _, _ => false
  | _, _ => false

theorem evalFunc_decl (sem : OpSem) (ctx : Ctx) (m : Module) (k : Nat) (c : String)
    (hk : 0 < k) (h : isDeclIn m c = true) : evalFunc sem ctx m k c = ctx.ext c := by
  obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  funext args
  simp only [isDeclIn] at h
  split at h
  · next f hf => simp [evalFunc, hf, h]
  · simp at h

theorem funcs_length_pos {m : Module} {name : String} {f : Func} (h : m.func? name = some f) :
    0 < m.funcs.length := by
  simp only [Module.func?] at h
  cases hfs : m.funcs with
  | nil => rw [hfs] at h; simp at h
  | cons _ _ => simp

theorem Corr.nil : Corr [] [] [] := fun x y h => by simp [Rho.find, List.lookup] at h

/-- C1's claim: the target function refines the source function wherever the source is modelled,
for every float model, every external environment, every index width and every input. -/
theorem c1Check_sound (i : C1Input) (h : c1Check i = true) :
    FuncRefinesC opSem (fun _ => True) (fun _ => True) i.tgt i.fn i.src i.fn := by
  intro ctx _ args _
  simp only [c1Check] at h
  split at h
  · next fS fT hfS hfT =>
    split at h
    · next argsS opsS argsT opsT hbS hbT =>
      simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, beq_iff_eq,
        List.all_eq_true] at h
      obtain ⟨⟨⟨⟨⟨hdS, hdT⟩, hty⟩, hfresh⟩, hdecl⟩, hrun⟩ := h
      obtain ⟨hnd, hfr⟩ := freshIds_sound hfresh
      have hl : argsS.length = argsT.length := by simpa using congrArg List.length hty
      -- unfold `denote` to the body evaluation on both sides
      have kS := funcs_length_pos hfS
      have kT := funcs_length_pos hfT
      obtain ⟨nS, hnS⟩ : ∃ n, i.src.funcs.length = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos kS).symm⟩
      obtain ⟨nT, hnT⟩ : ∃ n, i.tgt.funcs.length = n + 1 := ⟨_, (Nat.succ_pred_eq_of_pos kT).symm⟩
      simp only [denote, evalFunc, hfS, hfT, hdS, hdT, Bool.false_eq_true, ↓reduceIte, hbS, hbT,
        evalRegion, evalBlocks, evalBlock]
      apply OutLE.refinesC
      -- calls agree on every callee: each is a declaration in both modules
      have hcall : CallAgree (evalFunc opSem ctx i.src i.src.funcs.length)
          (evalFunc opSem ctx i.tgt i.tgt.funcs.length) (calleesOps opsS ++ calleesOps opsT) := by
        intro c hc
        have hb := hdecl c hc
        rw [evalFunc_decl _ _ _ _ _ kS hb.1, evalFunc_decl _ _ _ _ _ kT hb.2]
      by_cases hk : argsS.length = args.length
      · rw [bindAll_eq_ok hk, bindAll_eq_ok (hl ▸ hk)]
        simp only [Outcome.bind]
        exact runScript_sound _ _ ctx _ i.script opsS opsT _ _ hrun
          (by simpa using Corr.extend Corr.nil _ _ args (by simpa using hk)
                (by simpa [← hl] using hk) hnd (fun y hy p hp => by simp at hp))
          hcall
      · rw [bindAll_eq_stuck hk]
        exact Or.inr (Or.inl ⟨_, rfl⟩)
    · simp at h
  · simp at h

end MerlinVerify.Check
