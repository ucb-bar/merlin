/- Candidate-domain composition through the actual supplied suffix. The numerical and
per-function input restrictions are preserved rather than replaced by type-only assumptions. -/
import MerlinVerify.Check.PolicyW8A8NumericalQualified
import MerlinVerify.Check.PolicyW8A8BoundedChain

namespace MerlinVerify.Check.PolicyW8A8NumericalChain

open MerlinVerify Rules Definedness

theorem stripFacts {ib : Nat} : ∀ {facts : List (Refined.Fact BoundedFloatFacts.Info)} {args : List RVal},
    Refined.FactsHold PolicyW8A8NumericalFacts.model ib args
      (facts.map PolicyW8A8NumericalDefined.requireNonzero) → Refined.FactsHold BoundedFloatFacts.model ib args facts
  | [], _, holds => by cases holds; exact .nil
  | _ :: _, _, holds => by
    cases holds with
    | cons head tail => exact .cons ⟨head.1, head.2.1⟩ (stripFacts tail)

theorem inputDomain {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)} {args : List RVal}
    (domain : PolicyW8A8NumericalDefined.Domain ib types ranges args) :
    PolicyW8A8BoundedDefined.Domain ib types ranges args := by
  obtain ⟨facts, selected, holds⟩ := domain
  unfold PolicyW8A8NumericalDefined.inputFacts at selected
  cases decoded : PolicyW8A8BoundedDefined.inputFacts types ranges with
  | none => simp [decoded] at selected
  | some original =>
    simp only [decoded, Option.map_some, Option.some.injEq] at selected
    subst facts
    exact ⟨original, decoded, stripFacts holds⟩

def steps (ib : Nat) (intervals : PolicyW8A8NumericalQualified.InputIntervals) :
    PolicyFunctionChain.Steps (PolicyW8A8NumericalQualified.domains ib intervals) where
  check := PolicyW8A8DynamicChain.stepCheck ib
  sound checked := fun ctx permitted args domain =>
    PolicyW8A8DynamicChain.stepCheck_sound checked ctx
      ⟨permitted.1, PolicyW8A8NumericalContext.dynamic_context permitted.2⟩ args
      (PolicyW8A8BoundedChain.inputDomain (inputDomain domain))

def Correct (config : W8A8Reference.Config) (ib : Nat) (intervals : PolicyW8A8NumericalQualified.InputIntervals)
    (before after : Module) (rest : List Module) : Prop :=
  PolicyFunctionChain.Correct (PolicyW8A8NumericalQualified.domains ib intervals)
    (W8A8Reference.policy config) before after rest

def check (config : W8A8Reference.Config) (ib : Nat) (intervals : PolicyW8A8NumericalQualified.InputIntervals)
    (before after : Module) (rest : List Module) : Bool :=
  PolicyFunctionChain.check (steps ib intervals) (W8A8Reference.policy config) before after rest

theorem check_sound {config : W8A8Reference.Config} {ib : Nat}
    {intervals : PolicyW8A8NumericalQualified.InputIntervals} {before after : Module} {rest : List Module}
    (checked : check config ib intervals before after rest = true) : Correct config ib intervals before after rest :=
  PolicyFunctionChain.check_sound W8A8SourceAlpha.laws (steps ib intervals) checked

#print axioms inputDomain
#print axioms steps
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8NumericalChain
