/- Input-dependent neutral-element identities. A checked scalar constant supplies actual runtime
type evidence: successful arithmetic with it cannot silently alias an arbitrary tensor. -/
import MerlinVerify.Check.CorrLemmas
import MerlinVerify.Check.RulesArith
import MerlinVerify.Rules.Motion

namespace MerlinVerify.Rules.IntegerIdentity

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Kind where
  | add | sub | mul | and | or | xor
  deriving DecidableEq, Repr

def Kind.name : Kind → String
  | .add => "arith.addi" | .sub => "arith.subi" | .mul => "arith.muli"
  | .and => "arith.andi" | .or => "arith.ori" | .xor => "arith.xori"

def Kind.neutral (k : Kind) (w : Nat) : BitVec w :=
  match k with
  | .mul => 1
  | .and => BitVec.allOnes w
  | _ => 0

def Kind.reverseOK (k : Kind) (rev : Bool) : Prop := k ≠ .sub ∨ rev = false

instance (k : Kind) (rev : Bool) : Decidable (k.reverseOK rev) := by
  unfold Kind.reverseOK
  infer_instance

def ordered {α : Type} (rev : Bool) (a c : α) : List α :=
  if rev then [c, a] else [a, c]

/-- For every runtime value, including poison and malformed operands, a neutral scalar operation
either refuses that source execution or returns precisely its input. No type annotation is used
to assert the runtime input's width or scalar representation. -/
theorem meaning {op : Op} {r : Value} {w : Nat} (k : Kind) (rev : Bool)
    (hrev : k.reverseOK rev) (hd : dialectOf op.name = "arith")
    (hn : op.name = k.name) (hr : op.results = [r]) (ht : r.ty = .int w)
    (hm : unmodelledAttr? op.blind = none) (hf : overflowFlags op = [])
    (ctx : Ctx) (call : CallSem) (input : RVal) :
    OutLE (.ok [input]) (opSem ctx call op
      (ordered rev input (.scalar (.some (.bits w (k.neutral w))))) []) := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  have hf' : overflowFlags op.blind = [] := by
    simpa only [overflowFlags, attrEnum?, Op.blind_attr?] using hf
  cases input with
  | tensor t =>
    cases k <;> cases rev <;>
      simp_all [Kind.name, Kind.neutral, Kind.reverseOK, ordered, Arith.sem,
        Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results, liftEager, OutLE]
  | scalar value =>
    cases value with
    | poison =>
      cases k <;> cases rev <;>
        simp_all [Kind.name, Kind.neutral, Kind.reverseOK, ordered, Arith.sem,
          Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results, liftEager,
          Arith.intBin, OutLE]
    | some scalar =>
      cases scalar with
      | flt _ _ =>
        cases k <;> cases rev <;>
          simp_all [Kind.name, Kind.neutral, Kind.reverseOK, ordered, Arith.sem,
            Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results, liftEager,
            Arith.intBin, OutLE]
      | bits width value =>
        by_cases hw : width = w
        · subst width
          cases k <;> cases rev <;>
            simp_all [Kind.name, Kind.neutral, Kind.reverseOK, ordered, Arith.sem,
              Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results, liftEager,
              Arith.intBin, Arith.arithWithFlags, Arith.bv, OutLE]
        · cases k <;> cases rev <;>
            simp_all [Kind.name, Kind.neutral, Kind.reverseOK, ordered, Arith.sem,
              Arith.binOp, Arith.pureN, Arith.resultElt, Op.blind_results, liftEager,
              Arith.intBin, OutLE, Ne.symm hw]

def Checks (k : Kind) (rev : Bool) (ρ : Rho) (c b : Op) (rc rb : Value)
    (x : ValueId) (n : Int) (w : Nat) : Prop :=
  dialectOf c.name = "arith" ∧ c.name = "arith.constant" ∧
  unmodelledAttr? c.blind = none ∧ c.operands = [] ∧ c.regions = [] ∧ c.successors = [] ∧
  calleeOf c.blind = none ∧ c.attr? "value" = some (.int n (.int w)) ∧ rc.ty = .int w ∧
  BitVec.ofInt w n = k.neutral w ∧
  dialectOf b.name = "arith" ∧ b.name = k.name ∧ unmodelledAttr? b.blind = none ∧
  b.regions = [] ∧ b.successors = [] ∧ overflowFlags b = [] ∧ rb.ty = .int w ∧
  b.operands = ordered rev x rc.id ∧ k.reverseOK rev ∧
  ρ.find rc.id = none ∧ rc.id ≠ x

instance (k : Kind) (rev : Bool) (ρ : Rho) (c b : Op) (rc rb : Value)
    (x : ValueId) (n : Int) (w : Nat) : Decidable (Checks k rev ρ c b rc rb x n w) := by
  unfold Checks
  infer_instance

theorem nonterm (k : Kind) {op : Op} (h : op.name = k.name) : op.isTerminator = false := by
  cases k <;> simp [Op.isTerminator, terminatorNames, h, Kind.name]

theorem source_nonterms {k : Kind} {rev : Bool} {ρ : Rho} {c b : Op} {rc rb : Value}
    {x : ValueId} {n : Int} {w : Nat} (h : Checks k rev ρ c b rc rb x n w) :
    c.isTerminator = false ∧ b.isTerminator = false := by
  rcases h with ⟨_, hcn, _, _, _, _, _, _, _, _, _, hbn, _⟩
  exact ⟨by simp [Op.isTerminator, terminatorNames, hcn], nonterm k hbn⟩

theorem constant_sem {c : Op} {rc : Value} {n : Int} {w : Nat} {k : Kind}
    (hd : dialectOf c.name = "arith") (hn : c.name = "arith.constant")
    (hm : unmodelledAttr? c.blind = none) (hr : c.results = [rc])
    (ha : c.attr? "value" = some (.int n (.int w))) (hv : BitVec.ofInt w n = k.neutral w)
    (ctx : Ctx) (call : CallSem) :
    opSem ctx call c [] [] = .ok [.scalar (.some (.bits w (k.neutral w)))] := by
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  simp [Op.blind_name, Arith.sem, hn, Op.blind_attr?, ha,
    Op.blind_results, hr, scalarOfAttr, hv]

theorem arithmetic_success {b : Op} {rc rb : Value} {x : ValueId} {w : Nat}
    (k : Kind) (rev : Bool) (hrev : k.reverseOK rev)
    (hd : dialectOf b.name = "arith") (hn : b.name = k.name)
    (hr : b.results = [rb]) (ht : rb.ty = .int w)
    (hm : unmodelledAttr? b.blind = none) (hf : overflowFlags b = [])
    (ho : b.operands = ordered rev x rc.id) (hregs : b.regions = [])
    {ctx : Ctx} {call : CallSem} {env out : Env}
    (hc : env.get? rc.id = some (.scalar (.some (.bits w (k.neutral w)))))
    (hs : stepOp opSem ctx call env b = .ok out) :
    ∃ input, env.get? x = some input ∧ out = (rb.id, input) :: env := by
  unfold stepOp at hs
  rw [ho, hregs] at hs
  simp only [evalRegions] at hs
  cases hx : env.get? x with
  | none => cases rev <;> simp [ordered, Env.getAll, Outcome.mapM, Outcome.ofOption, hx, hc] at hs
  | some input =>
    have hargs : env.getAll (ordered rev x rc.id) =
        .ok (ordered rev input (.scalar (.some (.bits w (k.neutral w))))) := by
      cases rev <;> simp [ordered, Env.getAll, Outcome.mapM, Outcome.ofOption, hx, hc]
    rw [hargs, Outcome.bind_ok] at hs
    rcases meaning k rev hrev hd hn hr ht hm hf ctx call input with hu | ⟨reason, hu⟩ | he
    · rw [hu] at hs; cases hs
    · rw [hu] at hs; cases hs
    · rw [← he] at hs
      exact ⟨input, rfl, by simpa [Env.bindAll, hr] using hs.symm⟩

def targetPrefix (ρ : Rho) (c : Op) : List Op → Option Rho
  | [] => some ρ
  | [t] => alphaOp ρ c t
  | _ => none

theorem prefix_nonterms {ρ ρ' : Rho} {c : Op} {ts : List Op}
    (h : targetPrefix ρ c ts = some ρ') (hc : c.isTerminator = false) :
    ∀ t ∈ ts, t.isTerminator = false := by
  cases ts with
  | nil => simp
  | cons t rest =>
    cases rest with
    | cons _ _ => simp [targetPrefix] at h
    | nil =>
      have ha : alphaOp ρ c t = some ρ' := h
      simpa using (isTerminator_of_opHeadEq (alphaOp_spec ha).1).symm.trans hc

theorem prefix_sound {ρ ρ' : Rho} {c : Op} {rc : Value} {ts : List Op} {value : RVal}
    (hprefix : targetPrefix ρ c ts = some ρ')
    (hterm : c.isTerminator = false) (hregs : c.regions = [])
    (hcall : calleeOf c.blind = none) (hf : ρ.find rc.id = none)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (hc : Corr ρ envS envT)
    (hs : stepOp opSem ctx callS envS c = .ok ((rc.id, value) :: envS)) :
    (∀ t ∈ ts, t.isTerminator = false) ∧
    ∃ envT', execOps opSem ctx callT envT ts = .ok envT' ∧
      Corr ρ' ((rc.id, value) :: envS) envT' := by
  cases ts with
  | nil =>
    simp only [targetPrefix, Option.some.injEq] at hprefix
    subst ρ'
    exact ⟨by simp, envT, rfl, Corr.bindUnmapped hc hf value⟩
  | cons t rest =>
    cases rest with
    | cons _ _ => simp [targetPrefix] at hprefix
    | nil =>
      have hα : alphaOp ρ c t = some ρ' := hprefix
      have ht : t.isTerminator = false := (isTerminator_of_opHeadEq (alphaOp_spec hα).1) ▸ hterm
      obtain ⟨envT', hstep, hcorr⟩ := Motion.alpha_success hα hc hregs hcall hs
      refine ⟨by simpa using ht, envT', ?_, hcorr⟩
      rw [execOps_cons_step _ _ _ _ _ _ ht, hstep]
      rfl

theorem site_sound {k : Kind} {rev : Bool} {ρ ρ₁ : Rho} {c b : Op} {rc rb : Value}
    {x y : ValueId} {n : Int} {w : Nat} {ts : List Op}
    (hrc : c.results = [rc]) (hrb : b.results = [rb])
    (h : Checks k rev ρ c b rc rb x n w) (hp : targetPrefix ρ c ts = some ρ₁)
    (hxy : ρ₁.find x = some y)
    (ctx : Ctx) (callS callT : CallSem) (envS envT : Env) (hc : Corr ρ envS envT) :
    (∀ t ∈ ts, t.isTerminator = false) ∧
    SiteRel ((rb.id, y) :: ρ₁)
      (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS [c, b]) := by
  obtain ⟨hcd, hcn, hcm, hco, hcr, -, hctcall, hca, -, hcv, hbd, hbn, hbm, hbr,
    -, hbf, hbt, hbo, hrev, hhidden, -⟩ := h
  have hct : c.isTerminator = false := by simp [Op.isTerminator, terminatorNames, hcn]
  let value := RVal.scalar (.some (.bits w (k.neutral w)))
  have hstepC : stepOp opSem ctx callS envS c = .ok ((rc.id, value) :: envS) := by
    rw [Cse.closedStep hco hcr, constant_sem hcd hcn hcm hrc hca hcv]
    simp [Env.bindAll, hrc, value]
  obtain ⟨hts, envT', ht, hcorr⟩ := prefix_sound hp hct hcr hctcall hhidden
    ctx callS callT envS envT hc hstepC
  refine ⟨hts, ?_⟩
  rw [execOps_cons_step _ _ _ _ _ _ hct, hstepC, Outcome.bind_ok,
    execOps_cons_step _ _ _ _ _ _ (nonterm k hbn)]
  cases hb : stepOp opSem ctx callS ((rc.id, value) :: envS) b with
  | ub => exact .inl rfl
  | stuck reason => exact .inr (.inl ⟨reason, rfl⟩)
  | ok envB =>
    obtain ⟨input, hinput, rfl⟩ := arithmetic_success k rev hrev hbd hbn hrb hbt hbm hbf
      hbo hbr (by simp [Env.get?, List.lookup, value]) hb
    have htarget : envT'.get? y = some input := (hcorr x y hxy).symm.trans hinput
    exact .inr (.inr ⟨_, envT', rfl, ht, Cse.Corr.bindAlias hcorr htarget⟩)

def checkWith (k : Kind) (rev : Bool) (ρ : Rho) : List Op → List Op → Option Rho
  | [c, b], ts =>
    match c.results, b.results, c.attr? "value", b.operands with
    | [rc], [rb], some (.int n (.int w)), [a, z] =>
      let x := if rev then z else a
      if Checks k rev ρ c b rc rb x n w then do
        let ρ₁ ← targetPrefix ρ c ts
        let y ← ρ₁.find x
        some ((rb.id, y) :: ρ₁)
      else none
    | _, _, _, _ => none
  | _, _ => none

theorem checkWith_sound (k : Kind) (rev : Bool) :
    ∀ ρ ss ts ρ', checkWith k rev ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss)
  | ρ, [c, b], ts, ρ', h => by
    simp only [checkWith] at h
    split at h
    · next resultsC resultsB attr operands rc rb n w a z hrc hrb hca hbo =>
      generalize hxname : (if rev then z else a) = x at h
      split at h
      · next hchecks =>
        cases hp : targetPrefix ρ c ts with
        | none => simp [hp] at h
        | some ρ₁ =>
          simp only [hp, bind, Option.bind] at h
          cases hx : ρ₁.find x with
          | none => simp [hx] at h
          | some y =>
            simp only [hx, Option.some.injEq] at h
            subst ρ'
            obtain ⟨hct, hbt⟩ := source_nonterms hchecks
            refine ⟨by simp only [List.mem_cons, List.not_mem_nil, or_false]
                       intro op hm
                       rcases hm with rfl | rfl <;> assumption,
              prefix_nonterms hp hct, ?_⟩
            intro ctx callS callT envS envT hc _
            exact (site_sound hrc hrb hchecks hp hx ctx callS callT envS envT hc).2
      · cases h
    · cases h
  | _, [], _, _, h => by simp [checkWith] at h
  | _, [_], _, _, h => by simp [checkWith] at h
  | _, _ :: _ :: _ :: _, _, _, h => by simp [checkWith] at h

def candidates : List (Kind × Bool) :=
  [.add, .sub, .mul, .and, .or, .xor].flatMap fun k => [(k, false), (k, true)]

def check (ρ : Rho) (ss ts : List Op) : Option Rho :=
  candidates.findSome? fun (k, rev) => checkWith k rev ρ ss ts

theorem check_sound : ∀ ρ ss ts ρ', check ρ ss ts = some ρ' →
    (∀ op ∈ ss, op.isTerminator = false) ∧ (∀ op ∈ ts, op.isTerminator = false) ∧
    ∀ ctx callS callT envS envT, Corr ρ envS envT →
      CallAgree callS callT (calleesOps ss ++ calleesOps ts) →
      SiteRel ρ' (execOps opSem ctx callT envT ts) (execOps opSem ctx callS envS ss) := by
  intro ρ ss ts ρ' h
  obtain ⟨⟨k, rev⟩, _, hc⟩ := List.exists_of_findSome?_eq_some h
  exact checkWith_sound k rev ρ ss ts ρ' hc

def rule : Rule := ⟨"integer-neutral-element", check, check_sound⟩

end MerlinVerify.Rules.IntegerIdentity
