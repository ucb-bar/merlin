/- Exponent-aware W8A8 source analysis. All old numerical obligations remain
explicit; no primitive or transcendental interpretation is silently extended. -/
import MerlinVerify.Check.PolicyW8A8RotaryCast
import MerlinVerify.Check.PolicyW8A8RotaryDomain
import MerlinVerify.Check.PolicyW8A8RotaryTranspose
import MerlinVerify.Check.PolicyW8A8RotaryDequant
import MerlinVerify.Check.PolicyW8A8SingleWrite
import MerlinVerify.Check.PolicyW8A8NumericalCalled

namespace MerlinVerify.Check.PolicyW8A8RotaryDefined

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def scalars (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  (PolicyW8A8RotaryCast.book ib).orElse ((PolicyW8A8RotaryScalar.book ib).orElse
    ((PolicyW8A8RotaryDequant.book ib).orElse (Refined.adaptBook priorMap (PolicyW8A8NumericalDefined.book ib))))

def contracts (ib : Nat) : Contextual.Regions model sourceSem ib allowed :=
  Refined.adaptRegions priorMap (PolicyW8A8NumericalDefined.contracts ib)

def points (ib : Nat) (ranges : List Nat) :
    Contextual.Book model sourceSem ib (Contextual.PointConditions allowed ranges) :=
  Refined.adaptBook priorMap (PolicyW8A8NumericalDefined.points ib ranges)

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  let direct := PolicyW8A8SingleWrite.book elementwise PolicyW8A8NumericalContext.loopStable
    PolicyW8A8Defined.genericAgreement (.fromRegions (.operations (scalars ib)))
  let named := PolicyW8A8RotaryTranspose.book (.operations (scalars ib))
  let views := Contextual.Book.lift (PolicyW8A8ViewBook.book elementwise ib (invariants ib))
  let leaf := views.orElse (scalars ib)
  named.orElse (direct.orElse (Contextual.Structured.recursivePointBookWith (contracts ib) (points ib)
    elementwise PolicyW8A8NumericalContext.loopStable PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (invariants ib) leaf))

def validInputs (ib : Nat) (types : List VTy) (ranges : List (Option IndexRanges.Interval))
    (caps : List (Option Nat)) : Bool :=
  PolicyW8A8NumericalCalled.validInputs ib types ranges && PolicyW8A8RotaryDomain.check ib types ranges caps

#print axioms book

end MerlinVerify.Check.PolicyW8A8RotaryDefined
