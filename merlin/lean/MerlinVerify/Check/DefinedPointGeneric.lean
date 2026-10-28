/- Recursive structured inference supplies loop bounds only inside the corresponding
checked body. Numerical contracts remain independent of the body's current coordinates. -/
import MerlinVerify.Check.DefinedContextGeneric

namespace MerlinVerify.Check.Definedness.Contextual.Structured

open MerlinVerify Refined

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def nestedPointBookWith (contracts : Regions M sem ib allowed)
    (points : ∀ ranges, Book M sem ib (PointConditions allowed ranges))
    (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib) (leaf : Book M sem ib allowed) : Nat → Book M sem ib allowed
  | 0 => leaf
  | depth + 1 =>
    let nested := nestedPointBookWith contracts points elementwise stable agreement fillAgreement rules leaf depth
    let bodies := (PointRegions.fromRegions contracts).orElse (.operations fun ranges =>
      (points ranges).orElse (nested.restrict fun _ _ permitted => permitted.1))
    (bookPointRegions elementwise stable agreement rules bodies).orElse
      ((Fill.bookPointRegions elementwise stable fillAgreement bodies).orElse leaf)

/-- Derive recursion depth and iteration spaces from the actual operation. A point rule is
never available at function scope, where no structured iteration space has been established. -/
def recursivePointBookWith (contracts : Regions M sem ib allowed)
    (points : ∀ ranges, Book M sem ib (PointConditions allowed ranges))
    (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib) (leaf : Book M sem ib allowed) : Book M sem ib allowed where
  rule Γ op facts := (nestedPointBookWith contracts points elementwise stable agreement fillAgreement rules leaf
    (RegionDepth.op op)).rule Γ op facts
  sound Γ op facts results checked ctx call env args width permitted valid inputs :=
    (nestedPointBookWith contracts points elementwise stable agreement fillAgreement rules leaf
      (RegionDepth.op op)).sound Γ op facts results checked ctx call env args width permitted valid inputs

#print axioms recursivePointBookWith

end MerlinVerify.Check.Definedness.Contextual.Structured
