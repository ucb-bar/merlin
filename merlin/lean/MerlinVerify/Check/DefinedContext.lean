/- Initialized execution with explicit context/call conditions and value refinements.
The conditions are proof premises, never certificate flags. Function soundness must establish
them for the evaluator's actual call closure; arbitrary caller-supplied calls cannot qualify. -/
import MerlinVerify.Check.DefinedRefinements

namespace MerlinVerify.Check.Definedness.Contextual

open MerlinVerify Refined

abbrev Conditions := Ctx → CallSem → Prop

structure Book {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (allowed : Conditions) where
  rule : Refined.FEnv Info → Op → List (Refined.Fact Info) → Option (List (Refined.Fact Info))
  sound : ∀ Γ op fs rs, rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env → Refined.FactsHold M ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧ Refined.FactsHold M ib vs rs

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

/-- Existing unconditional rules keep their original obligations. -/
def Book.lift (R : Refined.Book M sem ib) : Book M sem ib allowed where
  rule := R.rule
  sound Γ op fs rs checked ctx call env args width _ valid inputs :=
    R.sound Γ op fs rs checked ctx call env args width valid inputs

def Book.orElse (first next : Book M sem ib allowed) : Book M sem ib allowed where
  rule Γ op fs := (first.rule Γ op fs).orElse fun _ => next.rule Γ op fs
  sound Γ op fs rs checked ctx call env args width permitted valid inputs := by
    cases selected : first.rule Γ op fs with
    | none =>
      simp only [selected, Option.orElse_none] at checked
      exact next.sound Γ op fs rs checked ctx call env args width permitted valid inputs
    | some results =>
      simp only [selected, Option.orElse_some, Option.some.injEq] at checked
      subst rs
      exact first.sound Γ op fs results selected ctx call env args width permitted valid inputs

/-- The entry facts define the input domain. No claimed fact is inferred from a sample or hint. -/
def Correct (M : Model Info) (sem : OpSem) (ib : Nat) (ctxOk : Ctx → Prop)
    (m : Module) (fn : String) (inputs : List (Refined.Fact Info)) : Prop :=
  FuncDefined sem (fun ctx => ctx.indexBits = ib ∧ ctxOk ctx)
    (fun args => Refined.FactsHold M ib args inputs) m fn

def checkOps (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) : List Op → List VTy → Bool
  | [], _ => false
  | op :: rest, yields =>
    match op.operands.mapM (fun id => Γ.lookup id) with
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

theorem checkOps_sound (R : Book M sem ib allowed) :
    ∀ Γ ops yields ctx call env, checkOps R Γ ops yields = true → Refined.EnvHolds M ib Γ env →
      ctx.indexBits = ib → allowed ctx call →
      ∃ vs, evalOps sem ctx call env ops = .ok vs ∧ HasTys ib vs yields ∧ ResultsDefined vs
  | _, [], _, _, _, _, checked, _, _, _ => by cases checked
  | Γ, (.mk name results operands attrs props regions successors) :: rest, yields,
      ctx, call, env, checked, valid, width, permitted => by
    simp only [checkOps] at checked
    split at checked
    · cases checked
    · next fs fetched =>
      obtain ⟨args, bound, inputs⟩ := valid.getAll operands fs fetched
      rw [evalOps.eq_2]
      split
      · next terminal =>
        rw [ite_eq_left terminal] at checked
        simp only [Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at checked
        obtain ⟨⟨rfl, initialized⟩, types⟩ := checked
        simp only [List.isEmpty_nil, ↓reduceIte, bound]
        exact ⟨args, rfl, types ▸ inputs.typed, inputs.defined (List.all_eq_true.1 initialized)⟩
      · next terminal =>
        rw [ite_eq_right terminal] at checked
        split at checked
        · split at checked
          · cases checked
          · next rs selected =>
            simp only [Bool.and_eq_true, beq_iff_eq] at checked
            obtain ⟨arity, tail⟩ := checked
            obtain ⟨vs, executed, holds⟩ :=
              R.sound Γ _ fs rs selected ctx call env args width permitted valid inputs
            simp only [Op.regions] at executed
            simp only [bound, Outcome.bind_ok, executed, Op.results]
            have length : results.length = vs.length :=
              arity.symm.trans (List.Forall₂.length_eq holds).symm
            rw [bindAll_ok length, Outcome.bind_ok]
            exact checkOps_sound R _ rest yields ctx call _ tail
              (valid.extend _ vs rs holds) width permitted
        · cases checked

def checkRegion (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) (region : Region)
    (inputs : List (Refined.Fact Info)) (yields : List VTy) : Bool :=
  match region with
  | .mk [.mk args ops] => args.mapM (fun arg => vtyOf ib arg.ty) == some (inputs.map (·.base.ty)) &&
      checkOps R ((args.map (·.id)).zip inputs ++ Γ) ops yields
  | _ => false

theorem checkRegion_sound (R : Book M sem ib allowed) {Γ : Refined.FEnv Info} {region : Region}
    {inputs : List (Refined.Fact Info)} {yields : List VTy}
    (checked : checkRegion R Γ region inputs yields = true) {ctx : Ctx} {call : CallSem} {env : Env}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call) (valid : Refined.EnvHolds M ib Γ env)
    {args : List RVal} (typed : Refined.FactsHold M ib args inputs) :
    ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧ HasTys ib vs yields ∧ ResultsDefined vs := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => simp [checkRegion] at checked
    | cons block rest =>
      cases rest with
      | cons _ _ => cases block; simp [checkRegion] at checked
      | nil =>
        cases block with
        | mk arguments ops =>
          simp only [checkRegion, Bool.and_eq_true, beq_iff_eq] at checked
          obtain ⟨types, body⟩ := checked
          have arity : arguments.length = args.length := by
            have length := optionMapM_length _ _ _ types
            rw [List.length_map] at length
            exact length.symm.trans (List.Forall₂.length_eq typed).symm
          simp only [evalRegion, evalBlocks, evalBlock]
          rw [bindAll_ok arity, Outcome.bind_ok]
          exact checkOps_sound R _ ops yields ctx call _ body
            (valid.extend _ args _ typed) width permitted

def check (R : Book M sem ib allowed) (m : Module) (fn : String)
    (inputs : List (Refined.Fact Info)) (results : List VTy) : Bool :=
  match m.func? fn with
  | some function => !function.isDecl && checkRegion R [] function.body inputs results
  | none => false

/-- The caller proves the condition for the actual module evaluator, including its call fuel.
For numerical-only conditions this is a context proof; call contracts can use the same interface. -/
theorem check_sound (R : Book M sem ib allowed) {m : Module} {fn : String}
    {inputs : List (Refined.Fact Info)} {results : List VTy} {ctxOk : Ctx → Prop}
    (checked : check R m fn inputs results = true)
    (compatible : ∀ ctx, ctx.indexBits = ib → ctxOk ctx →
      allowed ctx (evalFunc sem ctx m m.funcs.length)) : Correct M sem ib ctxOk m fn inputs := by
  unfold check at checked
  split at checked
  · next function found =>
    simp only [Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at checked
    obtain ⟨defined, body⟩ := checked
    intro ctx domain args inputsHold
    have empty : Refined.EnvHolds M ib [] [] := fun _ _ missing => by cases missing
    obtain ⟨vs, executed, _, initialized⟩ := checkRegion_sound R body domain.1
      (compatible ctx domain.1 domain.2) empty inputsHold
    refine ⟨vs, ?_, initialized⟩
    simpa only [denote, evalFunc, found, defined, Bool.false_eq_true, ↓reduceIte] using executed
  · cases checked

#print axioms Book.lift
#print axioms Book.orElse
#print axioms checkOps_sound
#print axioms checkRegion_sound
#print axioms check_sound

end MerlinVerify.Check.Definedness.Contextual
