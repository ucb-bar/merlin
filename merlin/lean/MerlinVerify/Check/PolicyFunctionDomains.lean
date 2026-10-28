/- Whole-module initialized policy qualification with explicit domains per function.
Shared argument types do not imply that two functions have the same input restrictions. -/
import MerlinVerify.Check.PolicyInitialized

namespace MerlinVerify.Check.PolicyFunctionDomains

open MerlinVerify PolicyModule

structure Domains (sem : OpSem) (ib : Nat) where
  contexts : Ctx → Prop
  inputs : String → List VTy → List RVal → Prop
  initialized : Module → String → List VTy → List VTy → Bool
  sound : ∀ m fn inputTypes outputTypes, initialized m fn inputTypes outputTypes = true →
    FuncDefined sem contexts (inputs fn inputTypes) m fn

variable {sem : OpSem} {ib : Nat}

def Correct (domains : Domains sem ib) (policy : Policy) (before after : Module) : Prop :=
  ∃ reference, policy.expand before = some reference ∧
    ∀ fn ∈ Module.definedNames reference, ∃ inputs outputs,
      TypedModule.signature? ib before fn = some (inputs, outputs) ∧
      TypedModule.signature? ib reference fn = some (inputs, outputs) ∧
      PolicyExact.Qualified sem domains.contexts (domains.inputs fn inputs) policy before after fn ∧
      FuncDefined sem domains.contexts (domains.inputs fn inputs) reference fn ∧
      FuncDefined sem domains.contexts (domains.inputs fn inputs) after fn

def domainCheck (domains : Domains sem ib) (before reference : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
  | some (inputs, outputs), some (otherInputs, otherOutputs) =>
    decide (inputs = otherInputs ∧ outputs = otherOutputs) &&
      domains.initialized before fn inputs outputs && domains.initialized reference fn inputs outputs
  | _, _ => false

def check (domains : Domains sem ib) (policy : Policy) (before after : Module) : Bool :=
  PolicyParametric.check policy before after &&
    match policy.expand before with
    | none => false
    | some reference => !(Module.definedNames reference).isEmpty &&
        (Module.definedNames reference).all (domainCheck domains before reference)

theorem check_sound (laws : ParametricAlpha.Laws sem) {domains : Domains sem ib}
    {policy : Policy} {before after : Module} (checked : check domains policy before after = true) :
    Correct domains policy before after := by
  simp only [check, Bool.and_eq_true] at checked
  obtain ⟨matched, defined⟩ := checked
  obtain ⟨actual, expansion, equal⟩ := PolicyParametric.check_sound laws matched
  cases expanded : policy.expand before with
  | none => simp [expanded] at defined
  | some reference =>
    have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
    subst actual
    simp only [expanded, Bool.and_eq_true, List.all_eq_true] at defined
    refine ⟨reference, expanded, ?_⟩
    intro fn member
    have supported := defined.2 fn member
    unfold domainCheck at supported
    split at supported
    · next inputs outputs otherInputs otherOutputs sourceSignature referenceSignature =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at supported
      obtain ⟨⟨⟨rfl, rfl⟩, sourceChecked⟩, referenceChecked⟩ := supported
      have sourceDefined := domains.sound _ _ _ _ sourceChecked
      have referenceDefined := domains.sound _ _ _ _ referenceChecked
      refine ⟨inputs, outputs, sourceSignature, referenceSignature,
        PolicyParametric.qualify_sound laws matched expanded sourceDefined referenceDefined.notStuck,
        referenceDefined, ?_⟩
      intro ctx permitted args domain
      rw [equal ctx fn args]
      exact referenceDefined ctx permitted args domain
    · cases supported

#print axioms check_sound

end MerlinVerify.Check.PolicyFunctionDomains
