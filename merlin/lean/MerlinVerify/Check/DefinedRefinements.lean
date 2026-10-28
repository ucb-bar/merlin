/- A reusable definedness walker with per-value refinement evidence. Existing operation books
are projected through their original type/initialization facts; additional families must prove
both successful execution and every new value fact. SSA rebinding shadows the old fact directly. -/
import MerlinVerify.Check.Defined

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify MerlinVerify.Check

structure Model (Info : Type) where
  holds : Info → RVal → Prop
  empty : Info
  empty_sound : ∀ v, holds empty v

structure Fact (Info : Type) where
  base : Definedness.Fact
  info : Info

variable {Info : Type} {M : Model Info}

def emptyFact (M : Model Info) (f : Definedness.Fact) : Fact Info := ⟨f, M.empty⟩

def Fact.Holds (M : Model Info) (ib : Nat) (v : RVal) (f : Fact Info) : Prop :=
  f.base.Holds ib v ∧ M.holds f.info v

def FactsHold (M : Model Info) (ib : Nat) (vs : List RVal) (fs : List (Fact Info)) : Prop :=
  List.Forall₂ (Fact.Holds M ib) vs fs

abbrev FEnv (Info : Type) := List (ValueId × Fact Info)

def EnvHolds (M : Model Info) (ib : Nat) (Γ : FEnv Info) (env : Env) : Prop :=
  ∀ x f, Γ.lookup x = some f → ∃ v, env.get? x = some v ∧ f.Holds M ib v

structure Book (M : Model Info) (sem : OpSem) (ib : Nat) where
  rule : FEnv Info → Op → List (Fact Info) → Option (List (Fact Info))
  sound : ∀ Γ op fs rs, rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧ FactsHold M ib vs rs

theorem FactsHold.base {ib : Nat} {vs : List RVal} {fs : List (Fact Info)}
    (h : FactsHold M ib vs fs) : Definedness.FactsHold ib vs (fs.map (·.base)) := by
  induction h with
  | nil => exact .nil
  | cons hv _ ih => exact .cons hv.1 ih

theorem FactsHold.typed {ib : Nat} {vs : List RVal} {fs : List (Fact Info)}
    (h : FactsHold M ib vs fs) : HasTys ib vs (fs.map (·.base.ty)) := by
  simpa [List.map_map, Function.comp_def] using h.base.typed

theorem FactsHold.defined {ib : Nat} {vs : List RVal} {fs : List (Fact Info)}
    (h : FactsHold M ib vs fs) (hi : ∀ f ∈ fs, f.base.initialized = true) : ResultsDefined vs := by
  apply h.base.defined
  intro f hf
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hf
  exact hi g hg

theorem FactsHold.empty {ib : Nat} {vs : List RVal} {fs : List Definedness.Fact}
    (h : Definedness.FactsHold ib vs fs) (M : Model Info) :
    FactsHold M ib vs (fs.map (emptyFact M)) := by
  induction h with
  | nil => exact .nil
  | cons hv _ ih => exact .cons ⟨hv, M.empty_sound _⟩ ih

theorem inputFacts (M : Model Info) {ib : Nat} {vs : List RVal} {ts : List VTy}
    (ht : HasTys ib vs ts) (hd : ResultsDefined vs) :
    FactsHold M ib vs (ts.map fun t => emptyFact M ⟨t, true⟩) := by
  simpa [List.map_map, Function.comp_def] using FactsHold.empty (Definedness.inputFacts ht hd) M

/-- Forget additional evidence when delegating to an existing proved operation book. -/
def baseEnv (Γ : FEnv Info) : Definedness.FEnv := Γ.map fun (x, f) => (x, f.base)

theorem baseEnv_lookup (Γ : FEnv Info) (x : ValueId) :
    (baseEnv Γ).lookup x = (Γ.lookup x).map (·.base) := by
  induction Γ with
  | nil => rfl
  | cons p Γ ih =>
    obtain ⟨y, f⟩ := p
    simp only [baseEnv, List.map_cons, List.lookup]
    split
    · rfl
    · simpa only [baseEnv] using ih

theorem EnvHolds.base {ib : Nat} {Γ : FEnv Info} {env : Env} (h : EnvHolds M ib Γ env) :
    Definedness.EnvHolds ib (baseEnv Γ) env := by
  intro x f hx
  rw [baseEnv_lookup] at hx
  cases hf : Γ.lookup x with
  | none => simp [hf] at hx
  | some g =>
    simp only [hf, Option.map_some, Option.some.injEq] at hx
    subst f
    obtain ⟨v, hv, hg⟩ := h x g hf
    exact ⟨v, hv, hg.1⟩

def Book.lift (M : Model Info) {sem : OpSem} {ib : Nat} (R : Definedness.Book sem ib) : Book M sem ib where
  rule Γ op fs := (R.rule (baseEnv Γ) op (fs.map (·.base))).map (List.map (emptyFact M))
  sound Γ op fs rs h ctx call env args hi he hf := by
    cases hr : R.rule (baseEnv Γ) op (fs.map (·.base)) with
    | none => simp [hr] at h
    | some ys =>
      simp only [hr, Option.map_some, Option.some.injEq] at h
      subst rs
      obtain ⟨vs, hv, hy⟩ := R.sound _ op _ ys hr ctx call env args hi he.base hf.base
      exact ⟨vs, hv, FactsHold.empty hy M⟩

def Book.orElse {sem : OpSem} {ib : Nat} (first next : Book M sem ib) : Book M sem ib where
  rule Γ op fs := (first.rule Γ op fs).orElse fun _ => next.rule Γ op fs
  sound Γ op fs rs h ctx call env args hi he hf := by
    cases hr : first.rule Γ op fs with
    | none =>
      simp only [hr, Option.orElse_none] at h
      exact next.sound Γ op fs rs h ctx call env args hi he hf
    | some ys =>
      simp only [hr, Option.orElse_some, Option.some.injEq] at h
      subst rs
      exact first.sound Γ op fs ys hr ctx call env args hi he hf

theorem EnvHolds.getAll {ib : Nat} {Γ : FEnv Info} {env : Env} (h : EnvHolds M ib Γ env) :
    ∀ xs fs, xs.mapM (fun x => Γ.lookup x) = some fs →
      ∃ vs, env.getAll xs = .ok vs ∧ FactsHold M ib vs fs
  | [], fs, hm => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at hm
    subst hm
    exact ⟨[], rfl, List.Forall₂.nil⟩
  | x :: xs, fs, hm => by
    simp only [List.mapM_cons] at hm
    cases hx : Γ.lookup x with
    | none => simp [hx] at hm
    | some f =>
      cases hxs : xs.mapM (fun x => Γ.lookup x) with
      | none => simp [hx, hxs] at hm
      | some fs' =>
        simp only [hx, hxs, Option.pure_def, bind, Option.bind, Option.some.injEq] at hm
        subst hm
        obtain ⟨v, hv, hf⟩ := h x f hx
        obtain ⟨vs, hvs, hfs⟩ := h.getAll xs fs' hxs
        refine ⟨v :: vs, ?_, List.Forall₂.cons hf hfs⟩
        change Outcome.mapM _ (x :: xs) = _
        rw [Outcome.mapM.eq_2]
        simp only [hv, Outcome.ofOption, Outcome.bind_ok]
        change (env.getAll xs).bind _ = _
        rw [hvs, Outcome.bind_ok]

theorem EnvHolds.extend {ib : Nat} {Γ : FEnv Info} {env : Env} (h : EnvHolds M ib Γ env) :
    ∀ (ids : List ValueId) (vs : List RVal) (fs : List (Fact Info)),
      FactsHold M ib vs fs → EnvHolds M ib (ids.zip fs ++ Γ) (ids.zip vs ++ env)
  | [], _, _, _ => by simpa using h
  | _ :: _, [], [], _ => by simpa using h
  | x :: xs, v :: vs, f :: fs, hv => by
    obtain _ | ⟨hf, hr⟩ := hv
    have ih := h.extend xs vs fs hr
    intro y g hy
    simp only [List.zip_cons_cons, List.cons_append, List.lookup] at hy
    simp only [List.zip_cons_cons, List.cons_append, Env.get?, List.lookup]
    by_cases he : y = x
    · subst he
      simp only [beq_self_eq_true, Option.some.injEq] at hy ⊢
      subst hy
      exact ⟨v, rfl, hf⟩
    · have hb : (y == x) = false := by simpa using he
      simp only [hb] at hy ⊢
      exact ih y g hy
  | _ :: _, [], _ :: _, hv => by cases hv
  | _ :: _, _ :: _, [], hv => by cases hv

/-- Only terminators yielding fully initialized results may finish an accepted block. -/
def checkOps {sem : OpSem} {ib : Nat} (R : Book M sem ib) (Γ : FEnv Info) : List Op → List VTy → Bool
  | [], _ => false
  | op :: rest, yields =>
    match op.operands.mapM (fun x => Γ.lookup x) with
    | none => false
    | some fs =>
      if op.isTerminator then
        rest.isEmpty && fs.all (·.base.initialized) && fs.map (·.base.ty) == yields
      else if op.successors.isEmpty then
        match R.rule Γ op fs with
        | none => false
        | some rs => rs.length == op.results.length &&
          checkOps R ((op.results.map (·.id)).zip rs ++ Γ) rest yields
      else false

theorem checkOps_sound {sem : OpSem} {ib : Nat} (R : Book M sem ib) :
    ∀ Γ ops yields ctx call env, checkOps R Γ ops yields = true → EnvHolds M ib Γ env →
      ctx.indexBits = ib → ∃ vs, evalOps sem ctx call env ops = .ok vs ∧
        HasTys ib vs yields ∧ ResultsDefined vs
  | _, [], _, _, _, _, h, _, _ => by cases h
  | Γ, (.mk n results operands attrs props regions successors) :: rest, yields, ctx, call, env,
      h, hΓ, hib => by
    simp only [checkOps] at h
    split at h
    · cases h
    · next fs hfs =>
      obtain ⟨args, hargs, hfacts⟩ := hΓ.getAll operands fs hfs
      rw [evalOps.eq_2]
      split
      · next ht =>
        rw [ite_eq_left ht] at h
        simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at h
        obtain ⟨⟨rfl, hi⟩, hty⟩ := h
        simp only [List.isEmpty_nil, ↓reduceIte, hargs]
        exact ⟨args, rfl, hty ▸ hfacts.typed, hfacts.defined (List.all_eq_true.1 hi)⟩
      · next ht =>
        rw [ite_eq_right ht] at h
        split at h
        · next hempty =>
          split at h
          · cases h
          · next rs hrule =>
            simp only [Bool.and_eq_true, beq_iff_eq] at h
            obtain ⟨hlen, hrest⟩ := h
            obtain ⟨vs, hop, hv⟩ := R.sound Γ _ fs rs hrule ctx call env args hib hΓ hfacts
            simp only [Op.regions] at hop
            simp only [hargs, Outcome.bind_ok, hop, Op.results]
            have hb : results.length = vs.length := hlen.symm.trans (List.Forall₂.length_eq hv).symm
            rw [bindAll_ok hb, Outcome.bind_ok]
            exact checkOps_sound R _ rest yields ctx call _ hrest (hΓ.extend _ vs rs hv) hib
        · cases h

/-- Check one region under the actual enclosing facts. Block arguments shadow captured values,
just as they do in the evaluator. The body can use only captures present in `Γ`. -/
def checkRegion {sem : OpSem} {ib : Nat} (R : Book M sem ib) (Γ : FEnv Info)
    (region : Region) (args : List (Fact Info)) (yields : List VTy) : Bool :=
  match region with
  | .mk [.mk bargs ops] => bargs.mapM (fun a => vtyOf ib a.ty) == some (args.map (·.base.ty)) &&
    checkOps R ((bargs.map (·.id)).zip args ++ Γ) ops yields
  | _ => false

theorem checkRegion_sound {sem : OpSem} {ib : Nat} (R : Book M sem ib) {Γ : FEnv Info}
    {region : Region} {argFacts : List (Fact Info)} {yields : List VTy}
    (h : checkRegion R Γ region argFacts yields = true) {ctx : Ctx} {call : CallSem} {env : Env}
    (hib : ctx.indexBits = ib) (hΓ : EnvHolds M ib Γ env) {args : List RVal}
    (ha : FactsHold M ib args argFacts) : ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧
      HasTys ib vs yields ∧ ResultsDefined vs := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => simp [checkRegion] at h
    | cons b bs =>
      cases bs with
      | cons b' bs => cases b; simp [checkRegion] at h
      | nil =>
        cases b with
        | mk bargs ops =>
          simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at h
          obtain ⟨ht, hops⟩ := h
          have hlen : bargs.length = args.length := by
            have htlen := optionMapM_length _ _ _ ht
            rw [List.length_map] at htlen
            exact htlen.symm.trans (List.Forall₂.length_eq ha).symm
          simp only [evalRegion, evalBlocks, evalBlock]
          rw [bindAll_ok hlen, Outcome.bind_ok]
          exact checkOps_sound R _ ops yields ctx call _ hops (hΓ.extend _ args _ ha) hib

/-- Function arguments are explicitly required to be typed and nonpoison. -/
def check {sem : OpSem} {ib : Nat} (R : Book M sem ib) (m : Module) (f : String)
    (argTys resTys : List VTy) : Bool :=
  match m.func? f with
  | some fn => !fn.isDecl && checkRegion R [] fn.body (argTys.map fun t => emptyFact M ⟨t, true⟩) resTys
  | none => false

theorem check_sound {sem : OpSem} {ib : Nat} (R : Book M sem ib) {m : Module} {f : String}
    {argTys resTys : List VTy} (h : check R m f argTys resTys = true) :
    FuncDefined sem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := by
  unfold check at h
  split at h
  · next fn hfn =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    obtain ⟨hnd, hb⟩ := h
    intro ctx hib args ha
    have hnil : EnvHolds M ib [] [] := fun _ _ h => by cases h
    obtain ⟨vs, hv, _, hd⟩ := checkRegion_sound R hb hib hnil (inputFacts M ha.1 ha.2)
    refine ⟨vs, ?_, hd⟩
    simpa only [denote, evalFunc, hfn, hnd, Bool.false_eq_true, ↓reduceIte] using hv
  · cases h

end MerlinVerify.Check.Definedness.Refined
