/- Regression statements for the distinction between support, refinement, and definedness. -/
import MerlinVerify.Sem.Defined
import MerlinVerify.Check.Typing
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Tests.Semantics

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

namespace MerlinVerify.Tests.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Tests.Semantics

def accepts (args : List Value) (ops : List Op) (argTys resTys : List VTy) : Bool :=
  check (book 64) { ops := [func "f" args ops] } "f" argTys resTys

#guard accepts [v 0 i32] [ret [0]] [.scalar i32] [.scalar i32]
#guard accepts [v 0 (tensorTy [2] i32)] [ret [0]] [.tensor [2] i32] [.tensor [2] i32]
#guard accepts [] [mkOp "arith.constant" [v 0 i32] [] [("value", .int 7 i32)], ret [0]] [] [.scalar i32]
#guard !accepts [] [mkOp "arith.constant" [v 0 i32] [] [("value", .int 7 i8)], ret [0]] [] [.scalar i32]
#guard !accepts [] [ret [99]] [] [.scalar i32]
#guard !accepts [v 0 i8] [ret [0]] [.scalar i32] [.scalar i32]
#guard !accepts [v 0 i32] [ret [0], ret [0]] [.scalar i32] [.scalar i32]

def empty : Op := mkOp "tensor.empty" [v 1 (tensorTy [2] i32)] []
def splat : Op := mkOp "tensor.splat" [v 2 (tensorTy [2] i32)] [0]

-- A dead uninitialized allocation is harmless; returning it is refused.
#guard accepts [v 0 i32] [empty, ret [0]] [.scalar i32] [.scalar i32]
#guard !accepts [] [empty, ret [1]] [] [.tensor [2] i32]
#guard accepts [v 0 i32] [splat, ret [2]] [.scalar i32] [.tensor [2] i32]
-- Multiple outputs must all be initialized.
#guard !accepts [v 0 i32] [empty, ret [0, 1]] [.scalar i32] [.scalar i32, .tensor [2] i32]
#guard accepts [v 0 i32] [splat, ret [0, 2]] [.scalar i32] [.scalar i32, .tensor [2] i32]

-- Neither unknown attributes nor unknown operations acquire a definedness proof.
#guard !accepts [] [mkOp "arith.constant" [v 0 i32] []
  [("value", .int 7 i32), ("review.unknown", .unit)], ret [0]] [] [.scalar i32]
#guard !accepts [v 0 i32] [mkOp "arith.divsi" [v 1 i32] [0, 0], ret [1]] [.scalar i32] [.scalar i32]

end MerlinVerify.Tests.Definedness
