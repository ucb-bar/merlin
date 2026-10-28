/- Strong initialized qualification for generated dynamic W8A8 and the existing finite
source/call path. Both branches use the explicit, stronger dynamic context for this claim. -/
import MerlinVerify.Check.PolicyW8A8DynamicDefined
import MerlinVerify.Check.PolicyW8A8DynamicCalled
import MerlinVerify.Check.PolicyW8A8FiniteQualified
import MerlinVerify.Check.PolicyInitialized

namespace MerlinVerify.Check.PolicyW8A8DynamicQualified

open MerlinVerify Rules

local notation "sourceSem" => W8A8SourceSemantics.opSem

def contexts (ib : Nat) (ctx : Ctx) : Prop :=
  ctx.indexBits = ib ∧ PolicyW8A8DynamicContext.ContextOk ctx

def initializedCheck (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  PolicyW8A8DynamicDefined.check ib m fn inputs outputs ||
    PolicyW8A8DynamicCalled.check ib m fn inputs outputs ||
    PolicyW8A8FiniteQualified.initializedCheck ib m fn inputs outputs

theorem initializedCheck_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : initializedCheck ib m fn inputs outputs = true) : PolicyW8A8DynamicDefined.Correct ib m fn inputs := by
  simp only [initializedCheck, Bool.or_eq_true] at checked
  rcases checked with (dynamic | called) | finite
  · exact PolicyW8A8DynamicDefined.check_sound dynamic
  · exact PolicyW8A8DynamicCalled.check_sound called
  · have initialized := PolicyW8A8FiniteQualified.initializedCheck_sound finite
    intro ctx permitted args domain
    exact initialized ctx ⟨permitted.1, PolicyW8A8DynamicContext.source_context permitted.2⟩ args domain

def domains (ib : Nat) : PolicyInitialized.Domains sourceSem ib where
  contexts := contexts ib
  inputs := PolicyW8A8FiniteDefined.Domain ib
  initialized := initializedCheck ib
  sound _ _ _ _ checked := initializedCheck_sound checked

def Correct (config : W8A8Reference.Config) (ib : Nat) (before after : Module) : Prop :=
  PolicyInitialized.Correct (domains ib) (W8A8Reference.policy config) before after

def check (config : W8A8Reference.Config) (ib : Nat) (before after : Module) : Bool :=
  PolicyInitialized.check (domains ib) (W8A8Reference.policy config) before after

theorem check_sound {config : W8A8Reference.Config} {ib : Nat} {before after : Module}
    (checked : check config ib before after = true) : Correct config ib before after :=
  PolicyInitialized.check_sound W8A8SourceAlpha.laws checked

#print axioms initializedCheck_sound
#print axioms domains
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8DynamicQualified
