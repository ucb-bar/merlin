/- Regression statements for the distinction between support, refinement, and definedness. -/
import MerlinVerify.Sem.Defined
import MerlinVerify.Check.Typing

namespace MerlinVerify.Tests

open MerlinVerify

example : ¬ RVal.Defined (.scalar .poison) := by simp [RVal.Defined]

example (s : Scalar) : RVal.Defined (.scalar (.some s)) := by simp [RVal.Defined]

-- Poison is well-typed, so definedness must be a separate input requirement.
example : Check.RVal.HasTy 64 (.scalar .poison) (.scalar (.int 32)) := trivial

-- Refinement alone accepts a concrete replacement for poison.
example (s : Scalar) : RValRefines (.scalar (.some s)) (.scalar .poison) := Or.inl rfl

example : (Outcome.ub : Outcome (List RVal)).NotStuck := trivial

-- Definedness does not permit UB, even though progress does.
example : ¬ ∃ vs, (Outcome.ub : Outcome (List RVal)) = .ok vs ∧ ResultsDefined vs := by simp

end MerlinVerify.Tests
