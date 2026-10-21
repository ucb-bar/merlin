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

/-- Strengthening a proof's context/call precondition does not strengthen its output facts. -/
def Book.restrict {stronger : Conditions} (R : Book M sem ib allowed)
    (implies : ∀ ctx call, stronger ctx call → allowed ctx call) : Book M sem ib stronger where
  rule := R.rule
  sound Γ op fs rs checked ctx call env args width permitted valid inputs :=
    R.sound Γ op fs rs checked ctx call env args width (implies ctx call permitted) valid inputs

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

/-- Infer the actual terminator facts, preserving proved numerical information across calls.
Only fully initialized results may finish an accepted block. -/
def inferOps (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) :
    List Op → Option (List (Refined.Fact Info))
  | [] => none
  | (op@(.mk _ results operands _ _ _ successors)) :: rest => do
    let fs ← operands.mapM (fun id => Γ.lookup id)
    if op.isTerminator then
      if rest.isEmpty && fs.all (·.base.initialized) then some fs else none
    else if successors.isEmpty then
      let rs ← R.rule Γ op fs
      if rs.length == results.length then
        inferOps R ((results.map (·.id)).zip rs ++ Γ) rest
      else none
    else none

theorem inferOps_sound (R : Book M sem ib allowed) :
    ∀ Γ ops results ctx call env, inferOps R Γ ops = some results → Refined.EnvHolds M ib Γ env →
      ctx.indexBits = ib → allowed ctx call →
      ∃ vs, evalOps sem ctx call env ops = .ok vs ∧ Refined.FactsHold M ib vs results ∧ ResultsDefined vs
  | _, [], _, _, _, _, checked, _, _, _ => by cases checked
  | Γ, (.mk name results operands attrs props regions successors) :: rest, out,
      ctx, call, env, checked, valid, width, permitted => by
    simp only [inferOps] at checked
    cases fetched : operands.mapM (fun id => Γ.lookup id) with
    | none => simp [fetched] at checked
    | some fs =>
      simp only [fetched, bind, Option.bind] at checked
      obtain ⟨args, bound, inputs⟩ := valid.getAll operands fs fetched
      rw [evalOps.eq_2]
      split
      · next terminal =>
        rw [ite_eq_left terminal] at checked
        split at checked
        · next conditions =>
          simp only [Bool.and_eq_true, List.isEmpty_iff] at conditions
          obtain ⟨rfl, initialized⟩ := conditions
          cases checked
          simp only [List.isEmpty_nil, ↓reduceIte, bound]
          exact ⟨args, rfl, inputs, inputs.defined (List.all_eq_true.1 initialized)⟩
        · cases checked
      · next terminal =>
        rw [ite_eq_right terminal] at checked
        by_cases successorsEmpty : successors.isEmpty = true
        · rw [ite_eq_left successorsEmpty] at checked
          cases selected : R.rule Γ (.mk name results operands attrs props regions successors) fs with
          | none => simp [selected] at checked
          | some rs =>
            simp only [selected] at checked
            split at checked
            · next arity =>
              have length := eq_of_beq arity
              obtain ⟨vs, executed, holds⟩ :=
                R.sound Γ _ fs rs selected ctx call env args width permitted valid inputs
              simp only [Op.regions] at executed
              simp only [bound, Outcome.bind_ok, executed, Op.results]
              have count : results.length = vs.length :=
                length.symm.trans (List.Forall₂.length_eq holds).symm
              rw [bindAll_ok count, Outcome.bind_ok]
              exact inferOps_sound R _ rest out ctx call _ checked
                (valid.extend _ vs rs holds) width permitted
            · cases checked
        · rw [ite_eq_right successorsEmpty] at checked
          cases checked

def inferRegion (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) (region : Region)
    (inputs : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  match region with
  | .mk [.mk args ops] =>
    if args.mapM (fun arg => vtyOf ib arg.ty) == some (inputs.map (·.base.ty)) then
      inferOps R ((args.map (·.id)).zip inputs ++ Γ) ops
    else none
  | _ => none

theorem inferRegion_sound (R : Book M sem ib allowed) {Γ : Refined.FEnv Info} {region : Region}
    {inputs results : List (Refined.Fact Info)}
    (checked : inferRegion R Γ region inputs = some results) {ctx : Ctx} {call : CallSem} {env : Env}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call) (valid : Refined.EnvHolds M ib Γ env)
    {args : List RVal} (typed : Refined.FactsHold M ib args inputs) :
    ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧ Refined.FactsHold M ib vs results ∧ ResultsDefined vs := by
  cases region with
  | mk blocks =>
    cases blocks with
    | nil => simp [inferRegion] at checked
    | cons block rest =>
      cases rest with
      | cons _ _ => cases block; simp [inferRegion] at checked
      | nil =>
        cases block with
        | mk arguments ops =>
          simp only [inferRegion] at checked
          split at checked
          · next types =>
            have types := eq_of_beq types
            have arity : arguments.length = args.length := by
              have length := optionMapM_length _ _ _ types
              rw [List.length_map] at length
              exact length.symm.trans (List.Forall₂.length_eq typed).symm
            simp only [evalRegion, evalBlocks, evalBlock]
            rw [bindAll_ok arity, Outcome.bind_ok]
            exact inferOps_sound R _ ops results ctx call _ checked
              (valid.extend _ args _ typed) width permitted
          · cases checked

def checkOps (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) (ops : List Op) (yields : List VTy) : Bool :=
  match inferOps R Γ ops with
  | some results => results.map (·.base.ty) == yields
  | none => false

theorem checkOps_sound (R : Book M sem ib allowed) :
    ∀ Γ ops yields ctx call env, checkOps R Γ ops yields = true → Refined.EnvHolds M ib Γ env →
      ctx.indexBits = ib → allowed ctx call →
      ∃ vs, evalOps sem ctx call env ops = .ok vs ∧ HasTys ib vs yields ∧ ResultsDefined vs := by
  intro Γ ops yields ctx call env checked valid width permitted
  unfold checkOps at checked
  cases inferred : inferOps R Γ ops with
  | none => simp [inferred] at checked
  | some results =>
    have types : results.map (·.base.ty) = yields := by simpa [inferred] using checked
    obtain ⟨vs, executed, holds, initialized⟩ := inferOps_sound R Γ ops results ctx call env
      inferred valid width permitted
    exact ⟨vs, executed, types ▸ holds.typed, initialized⟩

def checkRegion (R : Book M sem ib allowed) (Γ : Refined.FEnv Info) (region : Region)
    (inputs : List (Refined.Fact Info)) (yields : List VTy) : Bool :=
  match inferRegion R Γ region inputs with
  | some results => results.map (·.base.ty) == yields
  | none => false

theorem checkRegion_sound (R : Book M sem ib allowed) {Γ : Refined.FEnv Info} {region : Region}
    {inputs : List (Refined.Fact Info)} {yields : List VTy}
    (checked : checkRegion R Γ region inputs yields = true) {ctx : Ctx} {call : CallSem} {env : Env}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call) (valid : Refined.EnvHolds M ib Γ env)
    {args : List RVal} (typed : Refined.FactsHold M ib args inputs) :
    ∃ vs, evalRegion sem ctx call env region args = .ok vs ∧ HasTys ib vs yields ∧ ResultsDefined vs := by
  unfold checkRegion at checked
  cases inferred : inferRegion R Γ region inputs with
  | none => simp [inferred] at checked
  | some results =>
    have types : results.map (·.base.ty) = yields := by simpa [inferred] using checked
    obtain ⟨vs, executed, holds, initialized⟩ := inferRegion_sound R inferred width permitted valid typed
    exact ⟨vs, executed, types ▸ holds.typed, initialized⟩

/-- Infer a checked defined function's postcondition without trusting its declared result types. -/
def inferFunction (R : Book M sem ib allowed) (m : Module) (fn : String)
    (inputs : List (Refined.Fact Info)) (outputs : List VTy) : Option (List (Refined.Fact Info)) := do
  let function ← m.func? fn
  if function.isDecl then none else
    let results ← inferRegion R [] function.body inputs
    if results.map (·.base.ty) == outputs then some results else none

theorem inferFunction_sound (R : Book M sem ib allowed) {m : Module} {fn : String}
    {inputs results : List (Refined.Fact Info)} {outputs : List VTy}
    (checked : inferFunction R m fn inputs outputs = some results)
    {ctx : Ctx} {fuel : Nat} (width : ctx.indexBits = ib)
    (permitted : allowed ctx (evalFunc sem ctx m fuel)) {args : List RVal}
    (typed : Refined.FactsHold M ib args inputs) :
    ∃ vs, evalFunc sem ctx m (fuel + 1) fn args = .ok vs ∧
      Refined.FactsHold M ib vs results ∧ ResultsDefined vs := by
  unfold inferFunction at checked
  cases found : m.func? fn with
  | none => simp [found] at checked
  | some function =>
    simp only [found, bind, Option.bind] at checked
    split at checked
    · cases checked
    · next defined =>
      cases inferred : inferRegion R [] function.body inputs with
      | none => simp [inferred] at checked
      | some actual =>
        simp only [inferred] at checked
        split at checked
        · cases checked
          have empty : Refined.EnvHolds M ib [] [] := fun _ _ missing => by cases missing
          obtain ⟨vs, executed, holds, initialized⟩ := inferRegion_sound R inferred width permitted empty typed
          refine ⟨vs, ?_, holds, initialized⟩
          simpa only [evalFunc, found, defined, Bool.false_eq_true, ↓reduceIte] using executed
        · cases checked

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
#print axioms Book.restrict
#print axioms Book.orElse
#print axioms inferOps_sound
#print axioms inferRegion_sound
#print axioms inferFunction_sound
#print axioms checkOps_sound
#print axioms checkRegion_sound
#print axioms check_sound

end MerlinVerify.Check.Definedness.Contextual
