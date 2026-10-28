/- Downstream composition on domains indexed by function name. Each edge proves
refinement under precisely the original function's context and input restrictions. -/
import MerlinVerify.Check.PolicyFunctionDomains

namespace MerlinVerify.Check.PolicyFunctionChain

open MerlinVerify Rules

variable {sem : OpSem} {ib : Nat} {domains : PolicyFunctionDomains.Domains sem ib}

structure Steps (domains : PolicyFunctionDomains.Domains sem ib) where
  check : List VTy → List VTy → String → Module → Module → Bool
  sound : ∀ {inputs outputs fn before after}, check inputs outputs fn before after = true →
    FuncRefinesC sem domains.contexts (domains.inputs fn inputs) after fn before fn

def Correct (domains : PolicyFunctionDomains.Domains sem ib) (policy : PolicyModule.Policy)
    (before after : Module) (rest : List Module) : Prop :=
  PolicyFunctionDomains.Correct domains policy before ((after :: rest).getLast (List.cons_ne_nil _ _))

def suffixCheck (steps : Steps domains) (inputs outputs : List VTy) (fn : String) : Module → List Module → Bool
  | _, [] => true
  | before, after :: rest => steps.check inputs outputs fn before after &&
      (outsideChanges before after).unverified.isEmpty && suffixCheck steps inputs outputs fn after rest

theorem suffixCheck_sound (steps : Steps domains) {inputs outputs : List VTy} {fn : String}
    {after : Module} {rest : List Module} (checked : suffixCheck steps inputs outputs fn after rest = true) :
    ChainRefinesC sem domains.contexts (domains.inputs fn inputs) fn (after :: rest) := by
  induction rest generalizing after with
  | nil => trivial
  | cons next rest ih =>
    simp only [suffixCheck, Bool.and_eq_true] at checked
    exact ⟨steps.sound checked.1.1, ih checked.2⟩

def check (steps : Steps domains) (policy : PolicyModule.Policy)
    (before after : Module) (rest : List Module) : Bool :=
  PolicyFunctionDomains.check domains policy before after &&
    match policy.expand before with
    | none => false
    | some reference => (Module.definedNames reference).all fun fn =>
      match TypedModule.signature? ib reference fn with
      | none => false
      | some (inputs, outputs) => suffixCheck steps inputs outputs fn after rest

theorem check_sound (laws : ParametricAlpha.Laws sem) (steps : Steps domains)
    {policy : PolicyModule.Policy} {before after : Module} {rest : List Module}
    (checked : check steps policy before after rest = true) : Correct domains policy before after rest := by
  simp only [check, Bool.and_eq_true] at checked
  obtain ⟨reference, expanded, qualified⟩ := PolicyFunctionDomains.check_sound laws checked.1
  simp only [expanded, List.all_eq_true] at checked
  refine ⟨reference, expanded, ?_⟩
  intro fn member
  obtain ⟨inputs, outputs, sourceSignature, referenceSignature, policyQualified, referenceDefined, _⟩ := qualified fn member
  obtain ⟨actual, expansion, sourceDefined, referenceSupported, refined⟩ := policyQualified
  have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
  subst actual
  have post := checked.2 fn member
  simp only [referenceSignature] at post
  have finalRefined := PolicyModule.refining_suffix refined (suffixCheck_sound steps post)
  exact ⟨inputs, outputs, sourceSignature, referenceSignature,
    ⟨reference, expanded, sourceDefined, referenceSupported, finalRefined⟩,
    referenceDefined, finalRefined.defined referenceDefined⟩

#print axioms suffixCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyFunctionChain
