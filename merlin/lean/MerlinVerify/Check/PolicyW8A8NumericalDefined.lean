/- Separately named finite/nonzero-f32 input candidate. The original bounded domain and
checker are unchanged. Every numerical fact is proved in actual source execution. -/
import MerlinVerify.Check.PolicyW8A8NumericalOps
import MerlinVerify.Check.PolicyW8A8NumericalSquare
import MerlinVerify.Check.PolicyW8A8NumericalExtract
import MerlinVerify.Check.PolicyW8A8NumericalQuantize
import MerlinVerify.Check.PolicyW8A8NumericalReduce
import MerlinVerify.Check.PolicyW8A8NumericalDequant
import MerlinVerify.Check.PolicyW8A8BoundedDefined

namespace MerlinVerify.Check.PolicyW8A8NumericalDefined

open MerlinVerify Rules Definedness PolicyW8A8NumericalFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def contracts (ib : Nat) : Contextual.Regions model sourceSem ib allowed :=
  (PolicyW8A8NumericalSquare.contracts ib).orElse
    ((Refined.adaptRegions priorMap
      (Refined.adaptRegions BoundedFloatFacts.floatMap (PolicyW8A8DynamicDefined.contracts ib))).restrict
        (fun _ _ permitted => PolicyW8A8NumericalContext.dynamic_context permitted))

def leafBook (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  let views := Contextual.Book.lift (PolicyW8A8ViewBook.book elementwise ib (invariants ib))
  let extracts := Contextual.Book.lift (PolicyW8A8NumericalExtract.book ib)
  -- Existing finite-only structured contracts retain their own checked invariants. Trying
  -- that complete book as a fallback avoids demanding the new sign facts from old proofs.
  let legacy := (Refined.adaptBook priorMap (PolicyW8A8BoundedDefined.book ib)).restrict
    (fun _ _ permitted => PolicyW8A8NumericalContext.dynamic_context permitted)
  let scalars := (PolicyW8A8NumericalOps.book ib).orElse legacy
  let reductions := PolicyW8A8NumericalReduce.book elementwise (invariants ib)
    ((contracts ib).orElse (.operations scalars))
  (PolicyW8A8NumericalQuantize.book ib).orElse
    ((PolicyW8A8NumericalDequant.book ib).orElse
      (reductions.orElse ((PolicyW8A8NumericalOps.book ib).orElse (views.orElse (extracts.orElse legacy)))))

def points (ib : Nat) (ranges : List Nat) :
    Contextual.Book model sourceSem ib (Contextual.PointConditions allowed ranges) :=
  Refined.adaptBook priorMap
    (BoundedFloatFacts.PointIndex.book PolicyW8A8BoundedDefined.indexAgreement ib ranges)

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  Contextual.Structured.recursivePointBookWith (contracts ib) (points ib)
    elementwise PolicyW8A8NumericalContext.loopStable PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (invariants ib) (leafBook ib)

def requireNonzero (fact : Refined.Fact BoundedFloatFacts.Info) : Refined.Fact Info :=
  let numerical := match fact.base.ty with
    | .scalar (.float .f32) | .tensor _ (.float .f32) => Class.nonzeroFinite
    | _ => Class.any
  ⟨fact.base, ⟨fact.info, numerical⟩⟩

def inputFacts (types : List VTy) (ranges : List (Option IndexRanges.Interval)) :
    Option (List (Refined.Fact Info)) :=
  (PolicyW8A8BoundedDefined.inputFacts types ranges).map (List.map requireNonzero)

def Domain (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval)) (args : List RVal) : Prop :=
  ∃ facts, inputFacts types ranges = some facts ∧ Refined.FactsHold model ib args facts

def Correct (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) : Prop :=
  FuncDefined sourceSem (fun ctx => ctx.indexBits = ib ∧ PolicyW8A8NumericalContext.ContextOk ctx)
    (Domain ib types ranges) m fn

def check (ib : Nat) (m : Module) (fn : String) (types outputs : List VTy)
    (ranges : List (Option IndexRanges.Interval)) : Bool :=
  match inputFacts types ranges with
  | none => false
  | some facts =>
    (types.zip ranges).all (fun (ty, range) => PolicyW8A8BoundedDefined.inputIntervalValid ib ty range) &&
      Contextual.check (book ib) m fn facts outputs

theorem check_sound {ib : Nat} {m : Module} {fn : String} {types outputs : List VTy}
    {ranges : List (Option IndexRanges.Interval)} (checked : check ib m fn types outputs ranges = true) :
    Correct ib m fn types ranges := by
  unfold check at checked
  cases decoded : inputFacts types ranges with
  | none => simp [decoded] at checked
  | some facts =>
    simp only [decoded, Bool.and_eq_true] at checked
    have initialized := Contextual.check_sound (book ib) checked.2 (fun _ _ permitted => permitted)
    intro ctx permitted args domain
    obtain ⟨actual, same, holds⟩ := domain
    have equal : actual = facts := Option.some.inj (same.symm.trans decoded)
    subst actual
    exact initialized ctx permitted args holds

#print axioms book
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8NumericalDefined
