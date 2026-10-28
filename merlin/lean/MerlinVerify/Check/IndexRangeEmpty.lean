/- Empty intervals describe no scalar value. They can legitimately describe the elements
of an empty tensor, and arithmetic must not turn that evidence into an unknown interval.
Function-entry validation separately refuses empty declared input intervals. -/
import MerlinVerify.Check.IndexRanges

namespace MerlinVerify.Check.Definedness.IndexRangeEmpty

open MerlinVerify IndexRanges

def check : Option Interval → Bool
  | none => false
  | some range => range.upper ≤ range.lower

theorem not_holds {range : Option Interval} (checked : check range = true) (value : RVal) :
    ¬ model.holds range value := by
  cases range with
  | none => cases checked
  | some range =>
    have empty : range.upper ≤ range.lower := by simpa only [check, decide_eq_true_eq] using checked
    intro holds
    obtain ⟨_, _, lower, upper⟩ := holds
    omega

#print axioms not_holds

end MerlinVerify.Check.Definedness.IndexRangeEmpty
