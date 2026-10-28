/- Complete-boundary candidate qualification. Source and reference initialization use the
same explicitly stronger numerical domain; structural correspondence covers the entire target. -/
import MerlinVerify.Check.PolicyW8A8NumericalCalled
import MerlinVerify.Check.PolicyFunctionDomains
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

namespace MerlinVerify.Check.PolicyW8A8NumericalQualified

open MerlinVerify Rules
abbrev InputIntervals := PolicyW8A8NumericalCalled.InputIntervals

def domains (ib : Nat) (intervals : InputIntervals) : PolicyFunctionDomains.Domains W8A8SourceSemantics.opSem ib where
  contexts ctx := ctx.indexBits = ib ∧ PolicyW8A8NumericalContext.ContextOk ctx
  inputs fn types := PolicyW8A8NumericalDefined.Domain ib types (intervals fn)
  initialized m fn inputs outputs := PolicyW8A8NumericalCalled.check ib intervals m fn inputs outputs
  sound _ _ _ _ checked := PolicyW8A8NumericalCalled.check_sound checked

def Correct (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals) (before after : Module) : Prop :=
  PolicyFunctionDomains.Correct (domains ib intervals) (W8A8Reference.policy config) before after

def check (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals) (before after : Module) : Bool :=
  PolicyFunctionDomains.check (domains ib intervals) (W8A8Reference.policy config) before after

theorem check_sound {config : W8A8Reference.Config} {ib : Nat} {intervals : InputIntervals} {before after : Module}
    (checked : check config ib intervals before after = true) : Correct config ib intervals before after :=
  PolicyFunctionDomains.check_sound W8A8SourceAlpha.laws checked

#print axioms domains
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8NumericalQualified
