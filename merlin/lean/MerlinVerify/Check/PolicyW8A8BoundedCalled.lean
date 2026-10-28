/- Bounded W8A8 call contracts are inferred from actual helper bodies. Argument intervals
are selected by the callee's name and checked at each call using proved fact implication. -/
import MerlinVerify.Check.PolicyW8A8BoundedDefined
import MerlinVerify.Check.PolicyW8A8Called

namespace MerlinVerify.Check.PolicyW8A8BoundedCalled

open MerlinVerify Rules Definedness BoundedFloatFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
abbrev InputIntervals := String → List (Option IndexRanges.Interval)
abbrev Contract := Calls.Contract Info

def allowed (ib : Nat) (known : List Contract) : Contextual.Conditions :=
  NamedCalls.allowed model ib PolicyW8A8DynamicContext.ContextOk known

def book (ib : Nat) (known : List Contract) : Contextual.Book model sourceSem ib (allowed ib known) :=
  let calls := Calls.book model ib PolicyW8A8Called.callAgreement BoundedFloatFacts.check
    (fun _ _ checked _ holds => BoundedFloatFacts.check_sound checked holds) known
  let leaf := (calls.restrict (fun _ _ permitted => permitted.2)).orElse
    ((PolicyW8A8BoundedDefined.leafBook ib).restrict (fun _ _ permitted => permitted.1))
  let contracts := (Refined.adaptRegions floatMap (PolicyW8A8DynamicDefined.contracts ib)).restrict
    (fun _ _ permitted => permitted.1)
  Contextual.Structured.recursivePointBookWith contracts (PointIndex.book PolicyW8A8BoundedDefined.indexAgreement ib)
    elementwise (fun _ _ _ permitted => permitted) PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (invariants ib) leaf

def arguments (ib : Nat) (intervals : InputIntervals) (fn : String) (types : List VTy) :
    Option (List (Refined.Fact Info)) :=
  if (types.zip (intervals fn)).all (fun (ty, range) => PolicyW8A8BoundedDefined.inputIntervalValid ib ty range) then
    PolicyW8A8BoundedDefined.inputFacts types (intervals fn)
  else none

def system (ib : Nat) (intervals : InputIntervals) :
    NamedCalls.System model sourceSem ib PolicyW8A8DynamicContext.ContextOk where
  arguments := arguments ib intervals
  implies := BoundedFloatFacts.check
  implies_sound _ _ checked _ holds := BoundedFloatFacts.check_sound checked holds
  book := book ib

def stage (ib : Nat) (intervals : InputIntervals) (m : Module) : Nat → List Contract :=
  NamedCalls.stage (system ib intervals) m

theorem stage_sound (ib : Nat) (intervals : InputIntervals) (m : Module) : ∀ fuel ctx,
    ctx.indexBits = ib → PolicyW8A8DynamicContext.ContextOk ctx →
    Calls.Holds model ib (stage ib intervals m fuel) (evalFunc sourceSem ctx m fuel) :=
  NamedCalls.stage_sound (system ib intervals) m

def check (ib : Nat) (intervals : InputIntervals) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  (inputs.zip (intervals fn)).all (fun (ty, range) => PolicyW8A8BoundedDefined.inputIntervalValid ib ty range) &&
    NamedCalls.check (system ib intervals) m fn inputs outputs

theorem check_sound {ib : Nat} {intervals : InputIntervals} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib intervals m fn inputs outputs = true) :
    PolicyW8A8BoundedDefined.Correct ib m fn inputs (intervals fn) := by
  simp only [check, Bool.and_eq_true] at checked
  have initialized := NamedCalls.check_sound (system ib intervals) checked.2
  intro ctx permitted args domain
  obtain ⟨facts, selected, holds⟩ := domain
  apply initialized ctx permitted args
  refine ⟨facts, ?_, holds⟩
  simpa only [system, arguments, checked.1, ↓reduceIte] using selected

#print axioms book
#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8BoundedCalled
