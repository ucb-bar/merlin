/-
Straight-line definedness analysis. Facts track both the value type and whether every observable
element is initialized. A false initialization fact permits poison, but never permits UB or stuck
execution. Thus an uninitialized temporary can be fully overwritten without tainting the output.

Soundness obligation: acceptance implies `FuncDefined` on typed, nonpoison inputs. Every op rule
must prove successful execution and preservation of its output facts. Unsupported operations and region bodies
are refused; no verdict or certificate can supply a missing proof.
-/
import MerlinVerify.Check.Progress
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Check.Definedness

open MerlinVerify MerlinVerify.Check

structure Fact where
  ty : VTy
  initialized : Bool
  deriving DecidableEq

def Fact.Holds (ib : Nat) (v : RVal) (f : Fact) : Prop :=
  RVal.HasTy ib v f.ty ∧ (f.initialized = true → v.Defined)

def FactsHold (ib : Nat) (vs : List RVal) (fs : List Fact) : Prop :=
  List.Forall₂ (Fact.Holds ib) vs fs

abbrev FEnv := List (ValueId × Fact)

def EnvHolds (ib : Nat) (Γ : FEnv) (env : Env) : Prop :=
  ∀ x f, Γ.lookup x = some f → ∃ v, env.get? x = some v ∧ f.Holds ib v

/-- A total rule: even an output with `initialized = false` must be returned successfully.
Rules see the enclosing facts and must account for the actual region closures, including captures. -/
structure Book (sem : OpSem) (ib : Nat) where
  rule : FEnv → Op → List Fact → Option (List Fact)
  sound : ∀ Γ op fs rs, rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds ib Γ env → FactsHold ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧ FactsHold ib vs rs

theorem FactsHold.typed {ib : Nat} {vs : List RVal} {fs : List Fact} (h : FactsHold ib vs fs) :
    HasTys ib vs (fs.map (·.ty)) := by
  induction h with
  | nil => exact List.Forall₂.nil
  | cons hv _ ih => exact List.Forall₂.cons hv.1 ih

theorem FactsHold.defined {ib : Nat} {vs : List RVal} {fs : List Fact} (h : FactsHold ib vs fs)
    (hi : ∀ f ∈ fs, f.initialized = true) : ResultsDefined vs := by
  induction h with
  | nil => simp [ResultsDefined]
  | cons hv _ ih =>
    intro v hm
    rcases List.mem_cons.1 hm with rfl | hm
    · exact hv.2 (hi _ (List.mem_cons_self ..))
    · exact ih (fun f hf => hi f (List.mem_cons_of_mem _ hf)) v hm

theorem inputFacts {ib : Nat} {vs : List RVal} {ts : List VTy} (ht : HasTys ib vs ts)
    (hd : ResultsDefined vs) : FactsHold ib vs (ts.map fun t => ⟨t, true⟩) := by
  induction ht with
  | nil => exact List.Forall₂.nil
  | cons hv _ ih =>
    exact List.Forall₂.cons ⟨hv, fun _ => hd _ (List.mem_cons_self ..)⟩
      (ih (fun v hv => hd v (List.mem_cons_of_mem _ hv)))

theorem EnvHolds.getAll {ib : Nat} {Γ : FEnv} {env : Env} (h : EnvHolds ib Γ env) :
    ∀ xs fs, xs.mapM (fun x => Γ.lookup x) = some fs →
      ∃ vs, env.getAll xs = .ok vs ∧ FactsHold ib vs fs
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

theorem EnvHolds.extend {ib : Nat} {Γ : FEnv} {env : Env} (h : EnvHolds ib Γ env) :
    ∀ (ids : List ValueId) (vs : List RVal) (fs : List Fact),
      FactsHold ib vs fs → EnvHolds ib (ids.zip fs ++ Γ) (ids.zip vs ++ env)
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
def checkOps {sem : OpSem} {ib : Nat} (R : Book sem ib) (Γ : FEnv) : List Op → List VTy → Bool
  | [], _ => false
  | op :: rest, yields =>
    match op.operands.mapM (fun x => Γ.lookup x) with
    | none => false
    | some fs =>
      if op.isTerminator then
        rest.isEmpty && fs.all (·.initialized) && fs.map (·.ty) == yields
      else if op.successors.isEmpty then
        match R.rule Γ op fs with
        | none => false
        | some rs => rs.length == op.results.length &&
          checkOps R ((op.results.map (·.id)).zip rs ++ Γ) rest yields
      else false

theorem checkOps_sound {sem : OpSem} {ib : Nat} (R : Book sem ib) :
    ∀ Γ ops yields ctx call env, checkOps R Γ ops yields = true → EnvHolds ib Γ env →
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
def checkRegion {sem : OpSem} {ib : Nat} (R : Book sem ib) (Γ : FEnv)
    (region : Region) (args : List Fact) (yields : List VTy) : Bool :=
  match region with
  | .mk [.mk bargs ops] => bargs.mapM (fun a => vtyOf ib a.ty) == some (args.map (·.ty)) &&
    checkOps R ((bargs.map (·.id)).zip args ++ Γ) ops yields
  | _ => false

theorem checkRegion_sound {sem : OpSem} {ib : Nat} (R : Book sem ib) {Γ : FEnv}
    {region : Region} {argFacts : List Fact} {yields : List VTy}
    (h : checkRegion R Γ region argFacts yields = true) {ctx : Ctx} {call : CallSem} {env : Env}
    (hib : ctx.indexBits = ib) (hΓ : EnvHolds ib Γ env) {args : List RVal}
    (ha : FactsHold ib args argFacts) : ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧
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
def check {sem : OpSem} {ib : Nat} (R : Book sem ib) (m : Module) (f : String)
    (argTys resTys : List VTy) : Bool :=
  match m.func? f with
  | some fn => !fn.isDecl && checkRegion R [] fn.body (argTys.map fun t => ⟨t, true⟩) resTys
  | none => false

theorem check_sound {sem : OpSem} {ib : Nat} (R : Book sem ib) {m : Module} {f : String}
    {argTys resTys : List VTy} (h : check R m f argTys resTys = true) :
    FuncDefined sem (fun ctx => ctx.indexBits = ib)
      (fun args => HasTys ib args argTys ∧ ResultsDefined args) m f := by
  unfold check at h
  split at h
  · next fn hfn =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at h
    obtain ⟨hnd, hb⟩ := h
    intro ctx hib args ha
    have hnil : EnvHolds ib [] [] := fun _ _ h => by cases h
    obtain ⟨vs, hv, _, hd⟩ := checkRegion_sound R hb hib hnil (inputFacts ha.1 ha.2)
    refine ⟨vs, ?_, hd⟩
    simpa only [denote, evalFunc, hfn, hnd, Bool.false_eq_true, ↓reduceIte] using hv
  · cases h

end MerlinVerify.Check.Definedness
