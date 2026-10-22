/- Preserve declared integer input intervals throughout a checked W8A8 suffix. The
old suffix rules apply through a proved domain implication, without changing numerical laws. -/
import MerlinVerify.Check.PolicyFunctionChain
import MerlinVerify.Check.PolicyW8A8BoundedQualified
import MerlinVerify.Check.PolicyW8A8DynamicChain

namespace MerlinVerify.Check.PolicyW8A8BoundedChain
open MerlinVerify Rules Definedness

theorem inputFacts_finite {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    {facts : List (Refined.Fact BoundedFloatFacts.Info)} {args : List RVal}
    (selected : PolicyW8A8BoundedDefined.inputFacts types ranges = some facts)
    (holds : Refined.FactsHold BoundedFloatFacts.model ib args facts) :
    Refined.FactsHold FiniteFacts.model ib args (FiniteFacts.inputFacts types) := by
  induction types generalizing ranges facts args with
  | nil =>
    cases ranges with
    | nil => cases selected; cases holds; exact .nil
    | cons _ _ => cases selected
  | cons ty types ih =>
    cases ranges with
    | nil => cases selected
    | cons range ranges =>
      cases tail : PolicyW8A8BoundedDefined.inputFacts types ranges with
      | none => simp [PolicyW8A8BoundedDefined.inputFacts, tail] at selected
      | some rest =>
        simp only [PolicyW8A8BoundedDefined.inputFacts, tail, bind, Option.bind,
          Option.some.injEq] at selected
        subst facts
        cases holds with
        | cons head remaining => exact .cons ⟨head.1, head.2.1.1⟩ (ih tail remaining)

theorem inputDomain {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)} {args : List RVal}
    (domain : PolicyW8A8BoundedDefined.Domain ib types ranges args) :
    PolicyW8A8FiniteDefined.Domain ib types args := by
  obtain ⟨_, selected, holds⟩ := domain
  exact inputFacts_finite selected holds

def steps (ib : Nat) (intervals : PolicyW8A8BoundedQualified.InputIntervals) :
    PolicyFunctionChain.Steps (PolicyW8A8BoundedQualified.domains ib intervals) where
  check := PolicyW8A8DynamicChain.stepCheck ib
  sound checked := fun ctx permitted args domain =>
    PolicyW8A8DynamicChain.stepCheck_sound checked ctx permitted args (inputDomain domain)

def Correct (config : W8A8Reference.Config) (ib : Nat) (intervals : PolicyW8A8BoundedQualified.InputIntervals)
    (before after : Module) (rest : List Module) : Prop :=
  PolicyFunctionChain.Correct (PolicyW8A8BoundedQualified.domains ib intervals)
    (W8A8Reference.policy config) before after rest

def check (config : W8A8Reference.Config) (ib : Nat) (intervals : PolicyW8A8BoundedQualified.InputIntervals)
    (before after : Module) (rest : List Module) : Bool :=
  PolicyFunctionChain.check (steps ib intervals) (W8A8Reference.policy config) before after rest

theorem check_sound {config : W8A8Reference.Config} {ib : Nat}
    {intervals : PolicyW8A8BoundedQualified.InputIntervals} {before after : Module} {rest : List Module}
    (checked : check config ib intervals before after rest = true) : Correct config ib intervals before after rest :=
  PolicyFunctionChain.check_sound W8A8SourceAlpha.laws (steps ib intervals) checked

#print axioms inputFacts_finite
#print axioms inputDomain
#print axioms steps
#print axioms check_sound
end MerlinVerify.Check.PolicyW8A8BoundedChain
