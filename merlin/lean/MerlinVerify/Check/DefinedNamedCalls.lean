/- Close function-indexed call contracts at the evaluator's exact fuel. A function's
declared facts may depend on its name and may refuse unsupported input domains. -/
import MerlinVerify.Check.DefinedCallContracts
import MerlinVerify.Check.TypedModule

namespace MerlinVerify.Check.Definedness.NamedCalls

open MerlinVerify Refined

def allowed {Info : Type} (M : Model Info) (ib : Nat) (contexts : Ctx → Prop)
    (known : List (Calls.Contract Info)) (ctx : Ctx) (call : CallSem) : Prop :=
  contexts ctx ∧ Calls.Holds M ib known call

structure System {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (contexts : Ctx → Prop) where
  arguments : String → List VTy → Option (List (Refined.Fact Info))
  implies : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  implies_sound : ∀ actual required, implies actual required = true →
    ∀ values, Refined.FactsHold M ib values actual → Refined.FactsHold M ib values required
  book : ∀ known, Contextual.Book M sem ib (allowed M ib contexts known)

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {contexts : Ctx → Prop}

def Domain (system : System M sem ib contexts) (fn : String) (types : List VTy) (args : List RVal) : Prop :=
  ∃ facts, system.arguments fn types = some facts ∧ Refined.FactsHold M ib args facts

def stage (system : System M sem ib contexts) (m : Module) : Nat → List (Calls.Contract Info)
  | 0 => []
  | fuel + 1 =>
    let known := stage system m fuel
    (Module.definedNames m).filterMap fun fn => do
      let (inputs, outputs) ← TypedModule.signature? ib m fn
      let arguments ← system.arguments fn inputs
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
      cases declared : system.arguments fn inputs with
      | none => simp [declared] at accepted
      | some arguments =>
        simp only [declared, bind, Option.bind] at accepted
        cases inferred : Contextual.inferFunction (system.book (stage system m fuel)) m fn arguments outputs with
        | none => simp [inferred] at accepted
        | some results =>
          simp only [inferred, pure, Option.some.injEq] at accepted
          cases accepted
          exact Contextual.inferFunction_sound (system.book (stage system m fuel)) inferred width
            ⟨permitted, stage_sound system m fuel ctx width permitted⟩ inputsHold

def checkKnown (system : System M sem ib contexts) (known : List (Calls.Contract Info))
    (fn : String) (inputs outputs : List VTy) : Bool :=
  match Calls.find? known fn, system.arguments fn inputs with
  | some contract, some arguments => system.implies arguments contract.inputs &&
      contract.outputs.map (·.base.ty) == outputs
  | _, _ => false

def check (system : System M sem ib contexts) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  checkKnown system (stage system m (m.funcs.length + 1)) fn inputs outputs

theorem check_sound (system : System M sem ib contexts) {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check system m fn inputs outputs = true) :
    FuncDefined sem (fun ctx => ctx.indexBits = ib ∧ contexts ctx) (Domain system fn inputs) m fn := by
  unfold check checkKnown at checked
  split at checked
  · next contract arguments selected declared =>
    simp only [Bool.and_eq_true] at checked
    have member := List.mem_of_find?_eq_some selected
    have name : contract.name = fn := eq_of_beq
      (List.find?_some (p := fun c : Calls.Contract Info => c.name == fn) selected)
    intro ctx permitted args domain
    obtain ⟨actual, same, holds⟩ := domain
    have equal : actual = arguments := Option.some.inj (same.symm.trans declared)
    subst actual
    obtain ⟨values, executed, _holds, initialized⟩ := stage_sound system m (m.funcs.length + 1) ctx
      permitted.1 permitted.2 contract member args (system.implies_sound _ _ checked.1 _ holds)
    exact ⟨values, by simpa only [denote, name] using executed, initialized⟩
  · cases checked

#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.Definedness.NamedCalls
