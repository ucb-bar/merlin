/- Complete-boundary/suffix qualification under an explicitly constructed numerical family.
The old ten numerical laws and Agreement are now proved from this model rather than taken
as hypotheses. Input-domain, policy admission and native fidelity remain distinct. -/
import MerlinVerify.Check.PolicyW8A8IEEEContext
import MerlinVerify.Check.PolicyW8A8NumericalChain
import MerlinVerify.Check.PolicyW8A8RotaryDomain

namespace MerlinVerify.Check.PolicyW8A8IEEEQualified

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

def ModelContext (ctx : Ctx) : Prop :=
  ∃ fallback, ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)

theorem context_ok {ctx : Ctx} (model : ModelContext ctx) : PolicyW8A8NumericalContext.ContextOk ctx := by
  obtain ⟨fallback, same⟩ := model
  exact ⟨W8A8IEEEPrimitives.model fallback, W8A8IEEELaws.dynamic_laws fallback,
    W8A8IEEEPrimitives.agreement fallback, same⟩

def conditions (ib : Nat) (ctx : Ctx) : Prop := ctx.indexBits = ib ∧ ModelContext ctx

abbrev InputIntervals := PolicyW8A8NumericalQualified.InputIntervals

def domains (ib : Nat) (intervals : InputIntervals) : PolicyFunctionDomains.Domains sourceSem ib where
  contexts := conditions ib
  inputs fn types := PolicyW8A8NumericalDefined.Domain ib types (intervals fn)
  initialized := PolicyW8A8NumericalCalled.check ib intervals
  sound _ _ _ _ checked := fun ctx permitted args input =>
    PolicyW8A8NumericalCalled.check_sound checked ctx
      ⟨permitted.1, context_ok permitted.2⟩ args input

def steps (ib : Nat) (intervals : InputIntervals) : PolicyFunctionChain.Steps (domains ib intervals) where
  check := (PolicyW8A8NumericalChain.steps ib intervals).check
  sound checked := fun ctx permitted args input =>
    (PolicyW8A8NumericalChain.steps ib intervals).sound checked ctx
      ⟨permitted.1, context_ok permitted.2⟩ args input

def Correct (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals)
    (before after : Module) (suffix : List Module) : Prop :=
  PolicyFunctionChain.Correct (domains ib intervals) (W8A8Reference.policy config) before after suffix

def check (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals)
    (before after : Module) (suffix : List Module) : Bool :=
  PolicyFunctionChain.check (steps ib intervals) (W8A8Reference.policy config) before after suffix

theorem check_sound {config : W8A8Reference.Config} {ib : Nat} {intervals : InputIntervals}
    {before after : Module} {suffix : List Module}
    (checked : check config ib intervals before after suffix = true) :
    Correct config ib intervals before after suffix :=
  PolicyFunctionChain.check_sound W8A8SourceAlpha.laws (steps ib intervals) checked

theorem suffixCheck_eq_numerical (ib : Nat) (intervals : InputIntervals) (inputs outputs : List VTy)
    (fn : String) (after : Module) (suffix : List Module) :
    PolicyFunctionChain.suffixCheck (steps ib intervals) inputs outputs fn after suffix =
      PolicyFunctionChain.suffixCheck (PolicyW8A8NumericalChain.steps ib intervals)
        inputs outputs fn after suffix := by
  induction suffix generalizing after with
  | nil => rfl
  | cons next rest ih =>
    simp only [PolicyFunctionChain.suffixCheck, ih]
    rfl

theorem check_eq_numerical (config : W8A8Reference.Config) (ib : Nat) (intervals : InputIntervals)
    (before after : Module) (suffix : List Module) :
    check config ib intervals before after suffix =
      PolicyW8A8NumericalChain.check config ib intervals before after suffix := by
  simp only [check, PolicyW8A8NumericalChain.check, PolicyFunctionChain.check, suffixCheck_eq_numerical]
  rfl

/-- A checked input domain and the exact constructed context have a joint witness. This is
stronger than merely showing an unrelated syntactic type or law record is inhabited. -/
theorem domain_inhabited {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    {caps : List (Option Nat)} (checked : PolicyW8A8RotaryDomain.check ib types ranges caps = true) :
    ∃ ctx args, conditions ib ctx ∧ PolicyW8A8RotaryDomain.Domain ib types ranges caps args := by
  obtain ⟨args, inputs⟩ := PolicyW8A8RotaryDomain.check_inhabited checked
  refine ⟨PolicyW8A8IEEEContext.context W8A8IEEEPrimitives.unsupported
    (fun name _ => .stuck s!"uninterpreted W8A8 external {name}") ib, args, ?_, inputs⟩
  exact ⟨rfl, W8A8IEEEPrimitives.unsupported, rfl⟩

#print axioms check_sound
#print axioms check_eq_numerical
#print axioms domain_inhabited

end MerlinVerify.Check.PolicyW8A8IEEEQualified
