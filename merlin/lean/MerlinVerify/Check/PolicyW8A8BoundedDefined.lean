/- W8A8 execution with explicitly declared integer input intervals in addition to finite
f32 inputs. Bounds are part of this claim's domain, never inferred from a model name/sample. -/
import MerlinVerify.Check.PolicyW8A8DynamicDefined
import MerlinVerify.Check.BoundedFloatIndex
import MerlinVerify.Check.BoundedFloatExtract
import MerlinVerify.Check.DefinedPointGeneric
import MerlinVerify.Check.PolicyW8A8ViewBook

namespace MerlinVerify.Check.PolicyW8A8BoundedDefined

open MerlinVerify Rules Definedness BoundedFloatFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8DynamicContext.allowed

/-- These producer rules preserve scalar intervals. The numerical book handles floating
producers separately, so ordinary initialized execution cannot erase proved finite constants. -/
def intervalBook (ib : Nat) : Refined.Book IndexRanges.model sourceSem ib :=
  W8A8SourceBook.lift (((IndexRanges.constantBook ib).orElse (IndexArithmetic.book ib)).orElse
    ((IndexCastBounds.book ib).orElse ((IndexClampBounds.book ib).orElse (IndexRemainderBounds.book ib))))

theorem indexAgreement : PointIndex.Agreement sourceSem := by
  intro ctx call op args regions named
  apply W8A8SourceSemantics.unchanged
  simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]

def leafBook (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  (Contextual.Book.lift (PolicyW8A8ViewBook.book elementwise ib (invariants ib))).orElse
    ((Contextual.Book.lift (Extract.book ib)).orElse
      ((Refined.transportBook (toScalarRanges ib) (fromScalarRanges ib) (.lift (intervalBook ib))).orElse
        (Refined.adaptBook floatMap (PolicyW8A8DynamicDefined.leafBook ib))))

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  Contextual.Structured.recursivePointBookWith
    (Refined.adaptRegions floatMap (PolicyW8A8DynamicDefined.contracts ib)) (PointIndex.book indexAgreement ib)
    elementwise PolicyW8A8DynamicContext.loopStable PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (invariants ib) (leafBook ib)

/-- An interval is accepted only for an integer/index element type and a nonempty,
representable, nonnegative range. Its existence is a declared input restriction. -/
def inputIntervalValid (ib : Nat) (ty : VTy) (range : Option IndexRanges.Interval) : Bool :=
  match range with
  | none => true
  | some range =>
    let elt := match ty with | .scalar elt | .tensor _ elt => elt
    match IndexCastBounds.width? ib elt with
    | some width => 0 < width && range.lower < range.upper && range.upper ≤ 2 ^ (width - 1)
    | none => false

def inputFacts : List VTy → List (Option IndexRanges.Interval) → Option (List (Refined.Fact Info))
  | [], [] => some []
  | ty :: types, range :: ranges => do
    let rest ← inputFacts types ranges
    some (⟨⟨ty, true⟩, ⟨FloatFacts.fromClass (FiniteFacts.inputFact ty).info, range⟩⟩ :: rest)
  | _, _ => none

def check (ib : Nat) (m : Module) (fn : String) (types outputs : List VTy)
    (ranges : List (Option IndexRanges.Interval)) : Bool :=
  match inputFacts types ranges with
  | none => false
  | some facts =>
    (types.zip ranges).all (fun (ty, range) => inputIntervalValid ib ty range) &&
      Contextual.check (book ib) m fn facts outputs

def Domain (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval)) (args : List RVal) : Prop :=
  ∃ facts, inputFacts types ranges = some facts ∧ Refined.FactsHold model ib args facts

def Correct (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) : Prop :=
  FuncDefined sourceSem (fun ctx => ctx.indexBits = ib ∧ PolicyW8A8DynamicContext.ContextOk ctx)
    (Domain ib types ranges) m fn

theorem check_sound {ib : Nat} {m : Module} {fn : String} {types outputs : List VTy}
    {ranges : List (Option IndexRanges.Interval)} (checked : check ib m fn types outputs ranges = true) :
    Correct ib m fn types ranges := by
  unfold check at checked
  cases decoded : inputFacts types ranges with
  | none => simp [decoded] at checked
  | some facts =>
    simp only [decoded, Bool.and_eq_true, List.all_eq_true] at checked
    have initialized := Contextual.check_sound (book ib) checked.2 (fun _ _ permitted => permitted)
    intro ctx permitted args domain
    obtain ⟨actual, same, holds⟩ := domain
    have equal : actual = facts := Option.some.inj (same.symm.trans decoded)
    subst actual
    exact initialized ctx permitted args holds

#print axioms intervalBook
#print axioms indexAgreement
#print axioms book
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8BoundedDefined
