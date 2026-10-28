/- Strong policy qualification on the declared typed, initialized input domain. Independent
source and reference definedness are mandatory; support and matching alone cannot pass. The
recipe and semantic interpretation remain provisional, so this is not a reviewed receipt. -/
import MerlinVerify.Check.PolicyW8A8Defined
import MerlinVerify.Check.PolicyParametric
import MerlinVerify.Check.TypedModule
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

namespace MerlinVerify.Check.PolicyW8A8Qualified

open MerlinVerify Rules

local notation "sourceSem" => W8A8SourceSemantics.opSem

def Correct (config : W8A8Reference.Config) (ib : Nat) (before after : Module) : Prop :=
  ∃ reference, W8A8Reference.expand config before = some reference ∧
    ∀ fn ∈ Module.definedNames reference, ∃ argTys resTys,
      TypedModule.signature? ib before fn = some (argTys, resTys) ∧
      TypedModule.signature? ib reference fn = some (argTys, resTys) ∧
      PolicyExact.Qualified sourceSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args)
        (W8A8Reference.policy config) before after fn ∧
      FuncDefined sourceSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) reference fn ∧
      FuncDefined sourceSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) after fn

def domainCheck (ib : Nat) (before reference : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
  | some (args, results), some (otherArgs, otherResults) =>
    decide (args = otherArgs ∧ results = otherResults) &&
      PolicyW8A8Defined.check ib before fn args results &&
      PolicyW8A8Defined.check ib reference fn args results
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
    · next argTys resTys otherArgs otherResults sourceSignature referenceSignature =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at supported
      obtain ⟨⟨⟨rfl, rfl⟩, sourceChecked⟩, referenceChecked⟩ := supported
      have sourceDefined := PolicyW8A8Defined.check_sound sourceChecked
      have referenceDefined := PolicyW8A8Defined.check_sound referenceChecked
      refine ⟨argTys, resTys, sourceSignature, referenceSignature,
        PolicyParametric.qualify_sound W8A8SourceAlpha.laws matched expanded
          sourceDefined referenceDefined.notStuck, referenceDefined, ?_⟩
      intro ctx width args domain
      rw [equal ctx fn args]
      exact referenceDefined ctx width args domain
    · cases supported

#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Qualified
