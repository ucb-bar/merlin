/- Conditional initialized W8A8 qualification with declared input intervals per function.
The source, policy reference and actual output share exactly the same bounded input domain. -/
import MerlinVerify.Check.PolicyW8A8BoundedDefined
import MerlinVerify.Check.PolicyFunctionDomains
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

namespace MerlinVerify.Check.PolicyW8A8BoundedQualified

open MerlinVerify Rules

abbrev InputIntervals := String → List (Option Definedness.IndexRanges.Interval)

def domains (ib : Nat) (intervals : InputIntervals) : PolicyFunctionDomains.Domains W8A8SourceSemantics.opSem ib where
  contexts ctx := ctx.indexBits = ib ∧ PolicyW8A8DynamicContext.ContextOk ctx
  inputs fn types := PolicyW8A8BoundedDefined.Domain ib types (intervals fn)
  initialized m fn inputs outputs := PolicyW8A8BoundedDefined.check ib m fn inputs outputs (intervals fn)
  sound _ _ _ _ checked := PolicyW8A8BoundedDefined.check_sound checked

def Correct (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals) (before after : Module) : Prop :=
  PolicyFunctionDomains.Correct (domains ib intervals) (W8A8Reference.policy config) before after

def check (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals) (before after : Module) : Bool :=
  PolicyFunctionDomains.check (domains ib intervals) (W8A8Reference.policy config) before after

theorem check_sound {config : W8A8Reference.Config} {ib : Nat} {intervals : InputIntervals} {before after : Module}
    (checked : check config ib intervals before after = true) : Correct config ib intervals before after :=
  PolicyFunctionDomains.check_sound W8A8SourceAlpha.laws checked

#print axioms domains
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8BoundedQualified
