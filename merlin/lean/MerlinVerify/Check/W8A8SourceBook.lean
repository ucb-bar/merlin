/- Reuse an ordinary operation proof only after checking that the whole operation,
including its nested regions, has the same meaning under the additive W8A8 interpretation. -/
import MerlinVerify.Check.DefinedRefinements
import MerlinVerify.Rules.W8A8SourceFree

namespace MerlinVerify.Check.W8A8SourceBook

open MerlinVerify Rules Definedness

variable {Info : Type} {M : Refined.Model Info} {ib : Nat}

def lift (ordinary : Refined.Book M Sem.Ops.opSem ib) : Refined.Book M W8A8SourceSemantics.opSem ib where
  rule Γ op facts := if W8A8SourceFree.operation op then ordinary.rule Γ op facts else none
  sound Γ op facts results checked ctx call env args width valid inputs := by
    split at checked
    · next agreement =>
      simp only [W8A8SourceFree.operation, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at agreement
      rw [W8A8SourceFree.regions_eq op.regions call env agreement.2, W8A8SourceSemantics.unchanged agreement.1]
      exact ordinary.sound Γ op facts results checked ctx call env args width valid inputs
    · cases checked

#print axioms lift

end MerlinVerify.Check.W8A8SourceBook
