/- Conditional initialized execution through checked internal functions. Each stage derives
contracts from actual bodies using only the preceding stage's proved call contracts. Stage
fuel matches the frozen evaluator exactly; neither cycles nor external declarations are assumed. -/
import MerlinVerify.Check.DefinedCallContracts
import MerlinVerify.Check.FiniteImplication
import MerlinVerify.Check.PolicyW8A8FiniteDefined
import MerlinVerify.Check.TypedModule

namespace MerlinVerify.Check.PolicyW8A8Called

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem
abbrev Contract := Calls.Contract FiniteFacts.Class

def allowed (ib : Nat) (known : List Contract) (ctx : Ctx) (call : CallSem) : Prop :=
  PolicyW8A8FiniteQuantize.ContextOk ctx ∧ Calls.Holds FiniteFacts.model ib known call

theorem callAgreement : Calls.Agreement sourceSem := by
  intro ctx call op args regions named
  apply W8A8SourceSemantics.unchanged
  simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]

def book (ib : Nat) (known : List Contract) : Contextual.Book FiniteFacts.model sourceSem ib (allowed ib known) :=
  let calls := Calls.book FiniteFacts.model ib callAgreement FiniteFacts.Implication.check
    (fun _ _ checked _ inputs => FiniteFacts.Implication.check_sound checked inputs) known
  let leaf := (calls.restrict (fun _ _ permitted => permitted.2)).orElse
    ((PolicyW8A8FiniteDefined.leafBook ib).restrict (fun _ _ permitted => permitted.1))
  Contextual.Structured.recursiveBook FiniteFacts.Structured.elementwise
    (fun _ _ _ permitted => permitted) PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (FiniteFacts.Structured.invariants ib) leaf

def stage (ib : Nat) (m : Module) : Nat → List Contract
  | 0 => []
  | fuel + 1 =>
    let known := stage ib m fuel
    (Module.definedNames m).filterMap fun fn => do
      let (inputs, outputs) ← TypedModule.signature? ib m fn
      let arguments := FiniteFacts.inputFacts inputs
      let results ← Contextual.inferFunction (book ib known) m fn arguments outputs
      pure ⟨fn, arguments, results⟩

/-- Every contract is proved for the actual module evaluator at that stage's fuel. The base
stage has no contracts; successful circular assumptions cannot bootstrap each other. -/
theorem stage_sound (ib : Nat) (m : Module) : ∀ fuel ctx,
    ctx.indexBits = ib → PolicyW8A8FiniteQuantize.ContextOk ctx →
    Calls.Holds FiniteFacts.model ib (stage ib m fuel) (evalFunc sourceSem ctx m fuel)
  | 0, _, _, _ => by intro contract member; cases member
  | fuel + 1, ctx, width, numerical => by
    intro contract member args inputsHold
    obtain ⟨fn, _member, accepted⟩ := List.mem_filterMap.1 member
    cases signature : TypedModule.signature? ib m fn with
    | none => simp [signature] at accepted
    | some pair =>
      obtain ⟨inputs, outputs⟩ := pair
      simp only [signature, bind, Option.bind] at accepted
      cases inferred : Contextual.inferFunction (book ib (stage ib m fuel)) m fn
          (FiniteFacts.inputFacts inputs) outputs with
      | none => simp [inferred] at accepted
      | some results =>
        simp only [inferred, pure, Option.some.injEq] at accepted
        cases accepted
        exact Contextual.inferFunction_sound (book ib (stage ib m fuel)) inferred width
          ⟨numerical, stage_sound ib m fuel ctx width numerical⟩ inputsHold

def checkKnown (known : List Contract) (fn : String) (inputs outputs : List VTy) : Bool :=
  match Calls.find? known fn with
  | none => false
  | some contract => FiniteFacts.Implication.check (FiniteFacts.inputFacts inputs) contract.inputs &&
      contract.outputs.map (·.base.ty) == outputs

def check (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  checkKnown (stage ib m (m.funcs.length + 1)) fn inputs outputs

theorem check_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib m fn inputs outputs = true) : PolicyW8A8FiniteDefined.Correct ib m fn inputs := by
  unfold check checkKnown at checked
  cases selected : Calls.find? (stage ib m (m.funcs.length + 1)) fn with
  | none => simp [selected] at checked
  | some contract =>
    simp only [selected, Bool.and_eq_true] at checked
    have member := List.mem_of_find?_eq_some selected
    have name : contract.name = fn := eq_of_beq
      (List.find?_some (p := fun c : Contract => c.name == fn) selected)
    intro ctx permitted args inputsHold
    obtain ⟨vs, executed, _holds, initialized⟩ := stage_sound ib m (m.funcs.length + 1) ctx
      permitted.1 permitted.2 contract member args (FiniteFacts.Implication.check_sound checked.1 inputsHold)
    exact ⟨vs, by simpa only [denote, name] using executed, initialized⟩

#print axioms callAgreement
#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Called
