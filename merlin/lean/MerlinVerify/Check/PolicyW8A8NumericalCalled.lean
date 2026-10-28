/- Actual helper bodies supply numerical postconditions at the evaluator's call fuel.
Every function retains its own input intervals and finite/nonzero-f32 candidate domain. -/
import MerlinVerify.Check.PolicyW8A8NumericalInhabited
import MerlinVerify.Check.PolicyW8A8Called

namespace MerlinVerify.Check.PolicyW8A8NumericalCalled

open MerlinVerify Rules Definedness PolicyW8A8NumericalFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
abbrev InputIntervals := String → List (Option IndexRanges.Interval)
abbrev Contract := Calls.Contract Info

def allowed (ib : Nat) (known : List Contract) : Contextual.Conditions :=
  NamedCalls.allowed model ib PolicyW8A8NumericalContext.ContextOk known

def book (ib : Nat) (known : List Contract) : Contextual.Book model sourceSem ib (allowed ib known) :=
  let calls := Calls.book model ib PolicyW8A8Called.callAgreement PolicyW8A8NumericalFacts.check
    (fun _ _ checked _ holds => PolicyW8A8NumericalFacts.check_sound checked holds) known
  let leaf := (calls.restrict (fun _ _ permitted => permitted.2)).orElse
    ((PolicyW8A8NumericalDefined.leafBook ib).restrict (fun _ _ permitted => permitted.1))
  let contracts := (PolicyW8A8NumericalDefined.contracts ib).restrict (fun _ _ permitted => permitted.1)
  let points := fun ranges => Refined.adaptBook priorMap
    (BoundedFloatFacts.PointIndex.book PolicyW8A8BoundedDefined.indexAgreement ib ranges)
  Contextual.Structured.recursivePointBookWith contracts points elementwise (fun _ _ _ permitted => permitted)
    PolicyW8A8Defined.genericAgreement PolicyW8A8FiniteDefined.fillAgreement (invariants ib) leaf

def validInputs (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval)) : Bool :=
  (types.zip ranges).all (fun (ty, range) => PolicyW8A8BoundedDefined.inputIntervalValid ib ty range) &&
    PolicyW8A8NumericalInhabited.check ib types ranges

def arguments (ib : Nat) (intervals : InputIntervals) (fn : String) (types : List VTy) :
    Option (List (Refined.Fact Info)) :=
  if validInputs ib types (intervals fn) then PolicyW8A8NumericalDefined.inputFacts types (intervals fn) else none

def system (ib : Nat) (intervals : InputIntervals) :
    NamedCalls.System model sourceSem ib PolicyW8A8NumericalContext.ContextOk where
  arguments := arguments ib intervals
  implies := PolicyW8A8NumericalFacts.check
  implies_sound _ _ checked _ holds := PolicyW8A8NumericalFacts.check_sound checked holds
  book := book ib

def check (ib : Nat) (intervals : InputIntervals) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  validInputs ib inputs (intervals fn) && NamedCalls.check (system ib intervals) m fn inputs outputs

theorem check_sound {ib : Nat} {intervals : InputIntervals} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib intervals m fn inputs outputs = true) :
    PolicyW8A8NumericalDefined.Correct ib m fn inputs (intervals fn) := by
  simp only [check, Bool.and_eq_true] at checked
  have initialized := NamedCalls.check_sound (system ib intervals) checked.2
  intro ctx permitted args domain
  obtain ⟨facts, selected, holds⟩ := domain
  apply initialized ctx permitted args
  refine ⟨facts, ?_, holds⟩
  simpa only [system, arguments, checked.1, ↓reduceIte] using selected

theorem check_inhabited {ib : Nat} {intervals : InputIntervals} {m : Module} {fn : String}
    {inputs outputs : List VTy} (checked : check ib intervals m fn inputs outputs = true) :
    ∃ args, PolicyW8A8NumericalDefined.Domain ib inputs (intervals fn) args := by
  simp only [check, validInputs, Bool.and_eq_true] at checked
  exact PolicyW8A8NumericalInhabited.check_sound checked.1.2

#print axioms book
#print axioms check_sound
#print axioms check_inhabited

end MerlinVerify.Check.PolicyW8A8NumericalCalled
