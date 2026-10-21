/- Close conditional call contracts over actual function bodies at the evaluator's exact
fuel. The construction is generic in the numerical/refinement model and operation book. -/
import MerlinVerify.Check.DefinedCallContracts
import MerlinVerify.Check.TypedModule

namespace MerlinVerify.Check.Definedness.ClosedCalls

open MerlinVerify Refined

def allowed {Info : Type} (M : Model Info) (ib : Nat) (contexts : Ctx → Prop)
    (known : List (Calls.Contract Info)) (ctx : Ctx) (call : CallSem) : Prop :=
  contexts ctx ∧ Calls.Holds M ib known call

structure System {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (contexts : Ctx → Prop) where
  arguments : List VTy → List (Refined.Fact Info)
  implies : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  implies_sound : ∀ actual required, implies actual required = true →
    ∀ values, Refined.FactsHold M ib values actual → Refined.FactsHold M ib values required
  book : ∀ known, Contextual.Book M sem ib (allowed M ib contexts known)

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {contexts : Ctx → Prop}

def stage (system : System M sem ib contexts) (m : Module) : Nat → List (Calls.Contract Info)
  | 0 => []
  | fuel + 1 =>
    let known := stage system m fuel
    (Module.definedNames m).filterMap fun fn => do
      let (inputs, outputs) ← TypedModule.signature? ib m fn
      let arguments := system.arguments inputs
      let results ← Contextual.inferFunction (system.book known) m fn arguments outputs
      pure ⟨fn, arguments, results⟩

theorem stage_sound (system : System M sem ib contexts) (m : Module) : ∀ fuel ctx,
    ctx.indexBits = ib → contexts ctx → Calls.Holds M ib (stage system m fuel) (evalFunc sem ctx m fuel)
  | 0, _, _, _ => by intro contract member; cases member
  | fuel + 1, ctx, width, permitted => by
    intro contract member args inputsHold
    obtain ⟨fn, _member, accepted⟩ := List.mem_filterMap.1 member
    cases signature : TypedModule.signature? ib m fn with
    | none => simp [signature] at accepted
    | some pair =>
      obtain ⟨inputs, outputs⟩ := pair
      simp only [signature, bind, Option.bind] at accepted
      cases inferred : Contextual.inferFunction (system.book (stage system m fuel)) m fn
          (system.arguments inputs) outputs with
      | none => simp [inferred] at accepted
      | some results =>
        simp only [inferred, pure, Option.some.injEq] at accepted
        cases accepted
        exact Contextual.inferFunction_sound (system.book (stage system m fuel)) inferred width
          ⟨permitted, stage_sound system m fuel ctx width permitted⟩ inputsHold

def checkKnown (system : System M sem ib contexts) (known : List (Calls.Contract Info))
    (fn : String) (inputs outputs : List VTy) : Bool :=
  match Calls.find? known fn with
  | none => false
  | some contract => system.implies (system.arguments inputs) contract.inputs &&
      contract.outputs.map (·.base.ty) == outputs

def check (system : System M sem ib contexts) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  checkKnown system (stage system m (m.funcs.length + 1)) fn inputs outputs

theorem check_sound (system : System M sem ib contexts) {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check system m fn inputs outputs = true) :
    Contextual.Correct M sem ib contexts m fn (system.arguments inputs) := by
  unfold check checkKnown at checked
  cases selected : Calls.find? (stage system m (m.funcs.length + 1)) fn with
  | none => simp [selected] at checked
  | some contract =>
    simp only [selected, Bool.and_eq_true] at checked
    have member := List.mem_of_find?_eq_some selected
    have name : contract.name = fn := eq_of_beq
      (List.find?_some (p := fun c : Calls.Contract Info => c.name == fn) selected)
    intro ctx permitted args inputsHold
    obtain ⟨values, executed, _holds, initialized⟩ := stage_sound system m (m.funcs.length + 1) ctx
      permitted.1 permitted.2 contract member args (system.implies_sound _ _ checked.1 _ inputsHold)
    exact ⟨values, by simpa only [denote, name] using executed, initialized⟩

#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.Definedness.ClosedCalls
