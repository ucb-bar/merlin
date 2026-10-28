/- Rounded-real trig bounds inside actual structured regions. The old rules are reused
under a proved stronger context; no old checker or numerical interpretation is changed. -/
import MerlinVerify.Check.PolicyW8A8TrigScalar
import MerlinVerify.Check.PolicyW8A8AttentionScalar

namespace MerlinVerify.Check.PolicyW8A8TrigDefined

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8TrigScalar.allowed

theorem numerical_context {ctx : Ctx} {call : CallSem} (permitted : allowed ctx call) :
    PolicyW8A8NumericalContext.allowed ctx call :=
  PolicyW8A8IEEEQualified.context_ok (PolicyW8A8TrigScalar.original_allowed permitted)

theorem loopStable : Contextual.LoopStable allowed := by
  intro _ _ _ permitted
  exact permitted

def contracts (ib : Nat) : Contextual.Regions model sourceSem ib allowed :=
  (PolicyW8A8RotaryDefined.contracts ib).restrict (fun _ _ permitted => numerical_context permitted)

def points (ib : Nat) (ranges : List Nat) :
    Contextual.Book model sourceSem ib (Contextual.PointConditions allowed ranges) :=
  (PolicyW8A8RotaryDefined.points ib ranges).restrict
    (fun _ _ permitted => ⟨numerical_context permitted.1, permitted.2⟩)

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  let scalars := (PolicyW8A8AttentionScalar.book ib).orElse (PolicyW8A8TrigScalar.book ib)
  let direct := PolicyW8A8SingleWrite.book elementwise loopStable
    PolicyW8A8Defined.genericAgreement (.fromRegions (.operations scalars))
  direct.orElse (Contextual.Structured.recursivePointBookWith (contracts ib) (points ib)
    elementwise loopStable PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (invariants ib) scalars)

end MerlinVerify.Check.PolicyW8A8TrigDefined
