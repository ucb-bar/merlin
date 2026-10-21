/- Compose conditional initialized execution with source-derived policy validation. Source,
reference and actual output share one explicit numerical input/context domain. This is still
an unreviewed policy candidate, and does not install a public verified-pass receipt. -/
import MerlinVerify.Check.PolicyW8A8FiniteDefined
import MerlinVerify.Check.PolicyW8A8Called
import MerlinVerify.Check.PolicyParametric
import MerlinVerify.Check.TypedModule
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

namespace MerlinVerify.Check.PolicyW8A8FiniteQualified

open MerlinVerify Rules

local notation "sourceSem" => W8A8SourceSemantics.opSem

def contexts (ib : Nat) (ctx : Ctx) : Prop :=
  ctx.indexBits = ib ∧ PolicyW8A8FiniteQuantize.ContextOk ctx

def Correct (config : W8A8Reference.Config) (ib : Nat) (before after : Module) : Prop :=
  ∃ reference, W8A8Reference.expand config before = some reference ∧
    ∀ fn ∈ Module.definedNames reference, ∃ inputs outputs,
      TypedModule.signature? ib before fn = some (inputs, outputs) ∧
      TypedModule.signature? ib reference fn = some (inputs, outputs) ∧
      PolicyExact.Qualified sourceSem (contexts ib) (PolicyW8A8FiniteDefined.Domain ib inputs)
        (W8A8Reference.policy config) before after fn ∧
      FuncDefined sourceSem (contexts ib) (PolicyW8A8FiniteDefined.Domain ib inputs) reference fn ∧
      FuncDefined sourceSem (contexts ib) (PolicyW8A8FiniteDefined.Domain ib inputs) after fn

def initializedCheck (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  PolicyW8A8FiniteDefined.check ib m fn inputs outputs || PolicyW8A8Called.check ib m fn inputs outputs

theorem initializedCheck_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : initializedCheck ib m fn inputs outputs = true) : PolicyW8A8FiniteDefined.Correct ib m fn inputs := by
  simp only [initializedCheck, Bool.or_eq_true] at checked
  rcases checked with direct | called
  · exact PolicyW8A8FiniteDefined.check_sound direct
  · exact PolicyW8A8Called.check_sound called

def domainCheck (ib : Nat) (before reference : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
  | some (inputs, outputs), some (otherInputs, otherOutputs) =>
    decide (inputs = otherInputs ∧ outputs = otherOutputs) &&
      initializedCheck ib before fn inputs outputs && initializedCheck ib reference fn inputs outputs
  | _, _ => false

def check (config : W8A8Reference.Config) (ib : Nat) (before after : Module) : Bool :=
  PolicyParametric.check (W8A8Reference.policy config) before after &&
    match W8A8Reference.expand config before with
    | none => false
    | some reference => !(Module.definedNames reference).isEmpty &&
        (Module.definedNames reference).all (domainCheck ib before reference)

theorem check_sound {config : W8A8Reference.Config} {ib : Nat} {before after : Module}
    (checked : check config ib before after = true) : Correct config ib before after := by
  simp only [check, Bool.and_eq_true] at checked
  obtain ⟨matched, defined⟩ := checked
  obtain ⟨actual, expansion, equal⟩ := PolicyParametric.check_sound W8A8SourceAlpha.laws matched
  cases expanded : W8A8Reference.expand config before with
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
      have sourceDefined := initializedCheck_sound sourceChecked
      have referenceDefined := initializedCheck_sound referenceChecked
      refine ⟨inputs, outputs, sourceSignature, referenceSignature,
        PolicyParametric.qualify_sound W8A8SourceAlpha.laws matched expanded
          sourceDefined referenceDefined.notStuck, referenceDefined, ?_⟩
      intro ctx permitted args domain
      rw [equal ctx fn args]
      exact referenceDefined ctx permitted args domain
    · cases supported

#print axioms check_sound
#print axioms initializedCheck_sound

end MerlinVerify.Check.PolicyW8A8FiniteQualified
