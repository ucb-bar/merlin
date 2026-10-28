/- Exact normalization of fixed-width scalar cast chains. Runtime origin facts are learned from
successful typed source steps; result annotations alone never establish a cast identity. -/
import MerlinVerify.Check.BindingFacts
import MerlinVerify.Check.TypedNormalizer

namespace MerlinVerify.Rules.IntegerCast

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

inductive Kind where
  | zero | sign | trunc
  deriving DecidableEq, Repr

def Kind.name : Kind → String
  | .zero => "arith.extui" | .sign => "arith.extsi" | .trunc => "arith.trunci"

def Kind.widths (k : Kind) (a b : Nat) : Prop :=
  match k with
  | .zero | .sign => a < b
  | .trunc => b < a

instance (k : Kind) (a b : Nat) : Decidable (k.widths a b) := by
  cases k <;> unfold Kind.widths <;> infer_instance

def Kind.apply (k : Kind) (w : Nat) : Val → Val
  | .poison => .poison
  | .some (.bits _ x) => .some (.bits w (if k = .sign then x.signExtend w else x.setWidth w))
  | .some (.flt _ _) => .poison

/-- Only these four ordered pairs flatten without assumptions on the input bits. In particular,
zero-extending a signed extension cannot be replaced by extending the original input. -/
def Compatible (first second : Kind) (a b c : Nat) : Prop :=
  first.widths a b ∧ second.widths b c ∧
    ((first = .zero ∧ second = .zero) ∨ (first = .sign ∧ second = .sign) ∨
     (first = .zero ∧ second = .sign) ∨ (first = .trunc ∧ second = .trunc))

instance (first second : Kind) (a b c : Nat) : Decidable (Compatible first second a b c) := by
  unfold Compatible; infer_instance

/-- Poison is preserved exactly, and every initialized fixed-width scalar has the same final bits. -/
theorem meaning {first second : Kind} {a b c ib : Nat} {v : Val}
    (h : Compatible first second a b c) (ht : Val.HasTy ib (.int a) v) :
    second.apply c (first.apply b v) = first.apply c v := by
  rcases hasTy_intWidth (ib := ib) (t := .int a) rfl ht with rfl | ⟨x, rfl⟩
  · rfl
  · obtain ⟨hfirst, hsecond, hpairs⟩ := h
    rcases hpairs with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · simp only [Kind.widths] at hfirst hsecond
      simp [Kind.apply, BitVec.setWidth_setWidth (by omega : ¬ (b < a ∧ b < c))]
    · simp only [Kind.widths] at hfirst hsecond
      have hs : (x.signExtend b).signExtend c = x.signExtend c := by
        change BitVec.ofInt c (x.signExtend b).toInt = BitVec.ofInt c x.toInt
        rw [BitVec.toInt_signExtend_of_le (Nat.le_of_lt hfirst)]
      simp [Kind.apply, hs]
    · simp only [Kind.widths] at hfirst hsecond
      have hm : (x.setWidth b).msb = false := by
        simpa only [BitVec.setWidth'_eq] using (BitVec.msb_setWidth'_of_lt hfirst (x := x))
      simp [Kind.apply, BitVec.signExtend_eq_setWidth_of_msb_false hm,
        BitVec.setWidth_setWidth (by omega : ¬ (b < a ∧ b < c))]
    · simp only [Kind.widths] at hfirst hsecond
      simp [Kind.apply, BitVec.setWidth_setWidth_of_le x (Nat.le_of_lt hsecond)]

private def valueDecEq (a b : Value) : Decidable (a = b) :=
  decidable_of_iff (a.id = b.id ∧ a.ty = b.ty) (by cases a; cases b; simp)

local instance : DecidableEq Value := valueDecEq

theorem int_type_change {ib jb w : Nat} {v : Val} (h : Val.HasTy ib (.int w) v) :
    Val.HasTy jb (.int w) v := by
  cases v with
  | poison => trivial
  | some s => cases s <;> simp_all [Val.HasTy, Scalar.conforms]

structure Fact where
  root : ValueId
  result : Value
  inputWidth : Nat
  outputWidth : Nat
  kind : Kind
  deriving DecidableEq

/-- The fact includes the actual root value and its width, not just a producer's syntax. -/
def Holds (f : Fact) (env : Env) : Prop :=
  ∃ value, Val.HasTy 0 (.int f.inputWidth) value ∧
    env.get? f.root = some (.scalar value) ∧
    env.get? f.result.id = some (.scalar (f.kind.apply f.outputWidth value))

def model : BindingFacts.Model Fact where
  dependencies f := [f.root, f.result.id]
  holds := Holds
  congr f env env' he h := by
    obtain ⟨value, ht, hr, hv⟩ := h
    exact ⟨value, ht, (he f.root (by simp)).symm.trans hr,
      (he f.result.id (by simp)).symm.trans hv⟩

def Legal (Γ : TEnv) (s : Op) (f : Fact) : Prop :=
  dialectOf s.name = "arith" ∧ s.name = f.kind.name ∧ unmodelledAttr? s.blind = none ∧
  s.operands = [f.root] ∧ s.results = [f.result] ∧ f.result.ty = .int f.outputWidth ∧
  Γ.lookup f.root = some (.scalar (.int f.inputWidth)) ∧
  s.regions = [] ∧ s.successors = [] ∧ overflowFlags s = [] ∧ hasUnit s "nonNeg" = false ∧
  f.kind.widths f.inputWidth f.outputWidth

instance (Γ : TEnv) (s : Op) (f : Fact) : Decidable (Legal Γ s f) := by
  unfold Legal; infer_instance

/-- An unflagged legal scalar cast is total even for a poison input. -/
theorem cast_exec {Γ : TEnv} {s : Op} {f : Fact} (h : Legal Γ s f)
    (ctx : Ctx) (call : CallSem) (env : Env) {value : Val}
    (hv : env.get? f.root = some (.scalar value))
    (ht : Val.HasTy ctx.indexBits (.int f.inputWidth) value) :
    stepOp opSem ctx call env s =
      .ok ((f.result.id, .scalar (f.kind.apply f.outputWidth value)) :: env) := by
  obtain ⟨hd, hn, hm, ha, hr, hty, _, hregs, _, hf, hnn, hw⟩ := h
  have hargs : env.getAll s.operands = .ok [.scalar value] := by
    simp [ha, Env.getAll, Outcome.mapM, hv, Outcome.ofOption]
  have hres : Arith.resultElt s.blind = some (.int f.outputWidth) := by
    simp [Arith.resultElt, Op.blind_results, hr, hty]
  have hf' : overflowFlags s.blind = [] := by simpa [overflowFlags, attrEnum?] using hf
  have hnn' : hasUnit s.blind "nonNeg" = false := by simpa [hasUnit] using hnn
  unfold stepOp
  rw [hargs, Outcome.bind_ok, opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  rcases hasTy_intWidth (ib := ctx.indexBits) (t := .int f.inputWidth) rfl ht with
    rfl | ⟨x, rfl⟩
  all_goals
    cases hk : f.kind <;>
      simp_all [Kind.name, Kind.widths, Kind.apply, Arith.sem, Arith.widthOf,
        Ty.bitWidth?, Outcome.ofOption, Outcome.bindOp_eq, Arith.unOp, Arith.pureN,
        Arith.intUn, Arith.bv, liftEager, Env.bindAll]

/-- Parsing is merely a candidate generator: Legal is rechecked before a fact is used. -/
def candidate (Γ : TEnv) (s : Op) : Option Fact := do
  let kind ← if s.name = "arith.extui" then some Kind.zero
    else if s.name = "arith.extsi" then some Kind.sign
    else if s.name = "arith.trunci" then some Kind.trunc else none
  let ([root], [result]) := (s.operands, s.results) | none
  let some (.scalar (.int a)) := Γ.lookup root | none
  let .int b := result.ty | none
  pure ⟨root, result, a, b, kind⟩

def read (Γ : TEnv) (s : Op) : Option Fact := do
  let f ← candidate Γ s
  if Legal Γ s f then some f else none

theorem read_legal {Γ : TEnv} {s : Op} {f : Fact} (h : read Γ s = some f) :
    Legal Γ s f := by
  unfold read at h
  cases hc : candidate Γ s with
  | none => simp [hc] at h
  | some c =>
    simp only [hc, bind, Option.bind] at h
    split at h
    · cases h; assumption
    · cases h

/-- Facts that would retain a shadowed root are refused. Existing facts are independently
invalidated whenever any result binding is written. -/
def learn (Γ : TEnv) (s : Op) : Option Fact := do
  let f ← read Γ s
  if f.root ≠ f.result.id then some f else none

theorem learn_sound {ib : Nat} {Γ : TEnv} {s : Op} {f : Fact}
    (hl : learn Γ s = some f) {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hi : ctx.indexBits = ib)
    (hs : stepOp opSem ctx call env s = .ok env') : Holds f env' := by
  unfold learn at hl
  cases hr : read Γ s with
  | none => simp [hr] at hl
  | some item =>
    simp only [hr, bind, Option.bind] at hl
    split at hl
    · next hne =>
      cases hl
      have hc := read_legal hr
      obtain ⟨v, hv, ht⟩ := he _ _ hc.2.2.2.2.2.2.1
      cases v with
      | tensor _ => cases ht
      | scalar v =>
        change Val.HasTy ib (.int f.inputWidth) v at ht
        have ht' : Val.HasTy ctx.indexBits (.int f.inputWidth) v := by simpa [hi] using ht
        have hx := cast_exec hc ctx call env hv ht'
        rw [hx] at hs
        cases hs
        have hb : (f.root == f.result.id) = false := by simpa using hne
        exact ⟨v, int_type_change ht,
          by simpa [Env.get?, List.lookup, hb] using hv, by simp [Env.get?, List.lookup]⟩
    · cases hl

/-- The composed cast keeps the actual final result binding and its explicit result type. -/
def combined (first last : Fact) : Fact :=
  { last with root := first.root, inputWidth := first.inputWidth, kind := first.kind }

def direct (s : Op) (first : Fact) : Op :=
  .mk first.kind.name s.results [first.root] s.attrs s.props s.regions s.successors

def ReplacementChecks (Γ : TEnv) (facts : List Fact) (s : Op) (first last : Fact) : Prop :=
  Legal Γ s last ∧ first ∈ facts ∧ first.result.id = last.root ∧
  first.outputWidth = last.inputWidth ∧
  Compatible first.kind last.kind first.inputWidth first.outputWidth last.outputWidth ∧
  Legal Γ (direct s first) (combined first last)

instance (Γ : TEnv) (facts : List Fact) (s : Op) (first last : Fact) :
    Decidable (ReplacementChecks Γ facts s first last) := by
  unfold ReplacementChecks; infer_instance

/-- Equality of actual successful or failing steps. In particular no target poison, UB or stuck
outcome is hidden by the typed normalization used before C1 checking. -/
theorem replacement_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op} {first last : Fact}
    (h : ReplacementChecks Γ facts s first last) {ctx : Ctx} {call : CallSem} {env : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env)
    (hi : ctx.indexBits = ib) :
    stepOp opSem ctx call env (direct s first) = stepOp opSem ctx call env s := by
  obtain ⟨hs, hm, hid, hw, hc, ht⟩ := h
  obtain ⟨value, hvalue, hroot, hresult⟩ := hf first hm
  have hvalue' : Val.HasTy ctx.indexBits (.int first.inputWidth) value := int_type_change hvalue
  obtain ⟨actual, ha, hty⟩ := he _ _ hs.2.2.2.2.2.2.1
  have hav : actual = .scalar (first.kind.apply first.outputWidth value) := by
    rw [← hid, hresult] at ha
    exact (Option.some.inj ha).symm
  have hty' : Val.HasTy ctx.indexBits (.int last.inputWidth)
      (first.kind.apply first.outputWidth value) := by
    simpa [hav, RVal.HasTy, hi] using hty
  rw [cast_exec ht ctx call env hroot hvalue',
    cast_exec hs ctx call env (by simpa [hid] using hresult) hty']
  simp only [combined]
  rw [meaning hc hvalue']

/-- Candidate selection can be greedy; every semantic premise is checked independently below. -/
def proposal (Γ : TEnv) (facts : List Fact) (s : Op) : Option (Fact × Fact) := do
  let last ← read Γ s
  let first ← facts.find? fun f => f.result.id == last.root &&
    f.outputWidth == last.inputWidth &&
    decide (Compatible f.kind last.kind f.inputWidth f.outputWidth last.outputWidth)
  pure (first, last)

def rewrite (Γ : TEnv) (facts : List Fact) (s : Op) : Op :=
  match proposal Γ facts s with
  | some (first, last) =>
    if ReplacementChecks Γ facts s first last then direct s first else s
  | none => s

theorem rewrite_spec (Γ : TEnv) (facts : List Fact) (s : Op) :
    (rewrite Γ facts s).isTerminator = s.isTerminator ∧
    (rewrite Γ facts s).results = s.results ∧
    ∀ ib env, EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ∀ ctx call, ctx.indexBits = ib →
      stepOp opSem ctx call env (rewrite Γ facts s) = stepOp opSem ctx call env s := by
  unfold rewrite
  split
  · next first last hp =>
    split
    · next hc =>
      have hlegal := hc.1
      have hn : s.isTerminator = false := by
        have hn := hlegal.2.1
        cases hk : last.kind <;> simp [Op.isTerminator, terminatorNames, hn, Kind.name, hk]
      have hd : (direct s first).isTerminator = false := by
        cases hk : first.kind <;> simp [direct, Op.isTerminator, terminatorNames, Kind.name, hk, Op.name]
      exact ⟨hd.trans hn.symm, rfl, fun _ _ he hf _ _ hi => replacement_sound hc he hf hi⟩
    · exact ⟨rfl, rfl, fun _ _ _ _ _ _ _ => rfl⟩
  · exact ⟨rfl, rfl, fun _ _ _ _ _ _ _ => rfl⟩

def advance (Γ : TEnv) (facts : List Fact) (s : Op) : List Fact :=
  let retained := BindingFacts.forget model (s.results.map (·.id)) facts
  match learn Γ (rewrite Γ facts s) with
  | some f => f :: retained
  | none => retained

theorem advance_sound {ib : Nat} {Γ : TEnv} {facts : List Fact} {s : Op}
    {ctx : Ctx} {call : CallSem} {env env' : Env}
    (he : EnvTyped ib Γ env) (hf : BindingFacts.Valid model facts env)
    (hi : ctx.indexBits = ib) (hs : stepOp opSem ctx call env s = .ok env') :
    BindingFacts.Valid model (advance Γ facts s) env' := by
  have retained := BindingFacts.forget_step model hf hs
  unfold advance
  cases hl : learn Γ (rewrite Γ facts s) with
  | none => exact retained
  | some f =>
    have hstep : stepOp opSem ctx call env (rewrite Γ facts s) = .ok env' := by
      rw [(rewrite_spec Γ facts s).2.2 ib env he hf ctx call hi]; exact hs
    intro fact hfact
    rcases List.mem_cons.1 hfact with rfl | hm
    · exact learn_sound hl he hi hstep
    · exact retained _ hm

def normalize (ib : Nat) (Γ : TEnv) (facts : List Fact) : List Op → Option (List Op)
  | [] => some []
  | s :: ss =>
    if s.isTerminator then some (s :: ss) else do
      let _ ← s.operands.mapM (Γ.lookup ·)
      let rs ← s.results.mapM (fun r => vtyOf ib r.ty)
      let tail ← normalize ib ((s.results.map (·.id)).zip rs ++ Γ) (advance Γ facts s) ss
      pure (rewrite Γ facts s :: tail)

/-- The source-only traversal preserves the complete result outcome under actual runtime
argument types. Arbitrary surrounding checked operations may produce new typed SSA values. -/
theorem normalize_sound {ib : Nat} (R : Rulebook opSem ib) (d : Nat) :
    ∀ Γ facts ss ns yields ctx call env,
      checkOps R d Γ ss yields = true → normalize ib Γ facts ss = some ns →
      EnvTyped ib Γ env → BindingFacts.Valid model facts env →
      ctx.indexBits = ib → d ≤ ctx.loopIndex.length →
      evalOps opSem ctx call env ns = evalOps opSem ctx call env ss
  | _, _, [], _, _, _, _, _, h, _, _, _, _, _ => by simp [checkOps] at h
  | Γ, facts, s :: ss, ns, yields, ctx, call, env, h, hn, he, hf, hib, hd => by
    by_cases hterm : s.isTerminator = true
    · simp only [normalize, hterm, ↓reduceIte, Option.some.injEq] at hn
      subst ns; rfl
    · have ht : s.isTerminator = false := by simpa using hterm
      cases s with
      | mk name results operands attrs props regions successors =>
        simp only [checkOps] at h
        split at h
        · cases h
        · next tys hl =>
          rw [ite_eq_right hterm] at h
          split at h
          · cases h
          · next rs hr =>
            split at h
            · cases h
            · next sigs hR =>
              simp only [Bool.and_eq_true] at h
              obtain ⟨hregs, hrest⟩ := h
              simp only [normalize, ht, Bool.false_eq_true, ↓reduceIte, hl, hr,
                Op.operands, Op.results, bind, Option.bind] at hn
              cases htail : normalize ib ((results.map (·.id)).zip rs ++ Γ)
                  (advance Γ facts (.mk name results operands attrs props regions successors)) ss with
              | none => simp [htail] at hn
              | some tail =>
                simp only [htail, Option.pure_def, Option.some.injEq] at hn
                subst ns
                have hvterm := (rewrite_spec Γ facts
                    (.mk name results operands attrs props regions successors)).1.trans ht
                rw [evalOps_cons_step _ _ _ _ _ _ ht,
                  evalOps_cons_step _ _ _ _ _ _ hvterm,
                  (rewrite_spec Γ facts (.mk name results operands attrs props regions successors)).2.2
                    ib env he hf ctx call hib]
                cases hs : stepOp opSem ctx call env
                    (.mk name results operands attrs props regions successors) with
                | ub => rfl
                | stuck _ => rfl
                | ok env' =>
                  exact normalize_sound R d _ _ ss tail yields ctx call env' hrest htail
                    (step_envTyped R (op := .mk name results operands attrs props regions successors)
                      hl hr hR hregs hib hd he hs)
                    (advance_sound he hf hib hs) hib hd

def normalizer : TypedNormalizer where
  run ib _ Γ ss _ _ := normalize ib Γ [] ss
  sound ib d Γ ss hints ns yields ctx call env hc hn he hi hd :=
    normalize_sound (merlinBook ib) d Γ [] ss ns yields ctx call env hc hn he
      (by intro f hf; cases hf) hi hd

end MerlinVerify.Rules.IntegerCast
