/- Conditional dynamic W8A8 qualification through every actual downstream boundary.
The suffix preserves the same source interpretation, numerical premises and input domain.
Frozen-semantics checks are reusable only when both complete modules are source-free. -/
import MerlinVerify.Check.PolicyW8A8DynamicQualified
import MerlinVerify.Check.PolicyW8A8Chain
import MerlinVerify.Check.DeadModule

namespace MerlinVerify.Check.PolicyW8A8DynamicChain

open MerlinVerify Rules

local notation "sourceSem" => W8A8SourceSemantics.opSem

def Correct (config : W8A8Reference.Config) (ib : Nat)
    (before after : Module) (rest : List Module) : Prop :=
  PolicyW8A8DynamicQualified.Correct config ib before
    ((after :: rest).getLast (List.cons_ne_nil _ _))

theorem inputDomain {ib : Nat} {types : List VTy} {args : List RVal}
    (domain : PolicyW8A8FiniteDefined.Domain ib types args) :
    HasTys ib args types ∧ ResultsDefined args := by
  refine ⟨?_, domain.defined ?_⟩
  · simpa [FiniteFacts.inputFacts, FiniteFacts.inputFact, List.map_map,
      Function.comp_def] using domain.typed
  · intro fact member
    obtain ⟨ty, _, rfl⟩ := List.mem_map.1 member
    rfl

/-- SSA equivalence uses the additive interpretation directly. Other existing suffix rules
require checked agreement with the frozen interpretation on both entire modules. -/
def stepCheck (ib : Nat) (inputs outputs : List VTy) (fn : String) (before after : Module) : Bool :=
  ParametricModule.check before after ||
    (W8A8SourceFree.moduleCheck before && W8A8SourceFree.moduleCheck after &&
      (DeadModule.check ib before after || PolicyW8A8Chain.stepCheck ib inputs outputs fn before after))

theorem stepCheck_sound {ib : Nat} {inputs outputs : List VTy} {fn : String} {before after : Module}
    (checked : stepCheck ib inputs outputs fn before after = true) :
    FuncRefinesC sourceSem (PolicyW8A8DynamicQualified.contexts ib)
      (PolicyW8A8FiniteDefined.Domain ib inputs) after fn before fn := by
  simp only [stepCheck, Bool.or_eq_true, Bool.and_eq_true] at checked
  rcases checked with renamed | ⟨⟨sourceFree, targetFree⟩, frozen⟩
  · intro ctx _ args _
    rw [← ParametricModule.check_sound W8A8SourceAlpha.laws renamed ctx fn args]
    exact RefinesC.refl ResultsRefine.refl _
  · intro ctx permitted args domain
    rw [W8A8SourceFree.denote_eq sourceFree, W8A8SourceFree.denote_eq targetFree]
    rcases frozen with normalized | rewritten
    · rw [← DeadModule.check_sound normalized ctx permitted.1 fn args]
      exact RefinesC.refl ResultsRefine.refl _
    · exact PolicyW8A8Chain.stepCheck_sound rewritten ctx permitted.1 args (inputDomain domain)

def suffixCheck (ib : Nat) (inputs outputs : List VTy) (fn : String) : Module → List Module → Bool
  | _, [] => true
  | before, after :: rest => stepCheck ib inputs outputs fn before after &&
      (outsideChanges before after).unverified.isEmpty && suffixCheck ib inputs outputs fn after rest

theorem suffixCheck_sound {ib : Nat} {inputs outputs : List VTy} {fn : String}
    {after : Module} {rest : List Module}
    (checked : suffixCheck ib inputs outputs fn after rest = true) :
    ChainRefinesC sourceSem (PolicyW8A8DynamicQualified.contexts ib)
      (PolicyW8A8FiniteDefined.Domain ib inputs) fn (after :: rest) := by
  induction rest generalizing after with
  | nil => trivial
  | cons next rest ih =>
    simp only [suffixCheck, Bool.and_eq_true] at checked
    exact ⟨stepCheck_sound checked.1.1, ih checked.2⟩

def check (config : W8A8Reference.Config) (ib : Nat)
    (before after : Module) (rest : List Module) : Bool :=
  PolicyW8A8DynamicQualified.check config ib before after &&
    match W8A8Reference.expand config before with
    | none => false
    | some reference => (Module.definedNames reference).all fun fn =>
      match TypedModule.signature? ib reference fn with
      | none => false
      | some (inputs, outputs) => suffixCheck ib inputs outputs fn after rest

theorem check_sound {config : W8A8Reference.Config} {ib : Nat}
    {before after : Module} {rest : List Module}
    (checked : check config ib before after rest = true) : Correct config ib before after rest := by
  simp only [check, Bool.and_eq_true] at checked
  obtain ⟨reference, expanded, qualified⟩ := PolicyW8A8DynamicQualified.check_sound checked.1
  change W8A8Reference.expand config before = some reference at expanded
  simp only [expanded, List.all_eq_true] at checked
  refine ⟨reference, expanded, ?_⟩
  intro fn member
  obtain ⟨inputs, outputs, sourceSignature, referenceSignature, policyQualified, referenceDefined, _⟩ :=
    qualified fn member
  obtain ⟨actual, expansion, sourceDefined, referenceSupported, refined⟩ := policyQualified
  have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
  subst actual
  have post := checked.2 fn member
  simp only [referenceSignature] at post
  have finalRefined := PolicyModule.refining_suffix refined (suffixCheck_sound post)
  exact ⟨inputs, outputs, sourceSignature, referenceSignature,
    ⟨reference, expanded, sourceDefined, referenceSupported, finalRefined⟩,
    referenceDefined, finalRefined.defined referenceDefined⟩

#print axioms inputDomain
#print axioms stepCheck_sound
#print axioms suffixCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8DynamicChain
