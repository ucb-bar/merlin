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

def binary (name : String) (attrs : List (String × Attr) := []) : Bool :=
  accepts [v 0 i32, v 1 i32] [mkOp name [v 2 i32] [0, 1] attrs, ret [2]]
    [.scalar i32, .scalar i32] [.scalar i32]

#guard intBinaryNames.all binary
#guard !binary "arith.addi" [("overflowFlags", .enum "arith" "overflow" "nsw")]
#guard !binary "arith.muli" [("overflowFlags", .enum "arith" "overflow" "nuw")]
#guard !binary "arith.shli"
#guard !binary "arith.divui"
#guard !accepts [v 0 i32, v 1 i8] [mkOp "arith.addi" [v 2 i32] [0, 1], ret [2]]
  [.scalar i32, .scalar i8] [.scalar i32]
#guard !accepts [v 0 i32] [mkOp "arith.addi" [v 2 i8] [0, 0], ret [2]] [.scalar i32] [.scalar i8]

def reshape (name : String) (src dst : List Nat) (groups : List (List Nat)) : Bool :=
  accepts [v 0 (tensorTy src i32)]
    [mkOp name [v 1 (tensorTy dst i32)] [0]
      [("reassociation", .array (groups.map fun g => .array (g.map fun d => .int d (.int 64))))], ret [1]]
    [.tensor src i32] [.tensor dst i32]

#guard reshape "tensor.collapse_shape" [2, 3] [6] [[0, 1]]
#guard reshape "tensor.expand_shape" [6] [2, 3] [[0, 1]]
#guard reshape "tensor.collapse_shape" [0, 3] [0] [[0, 1]]
#guard !reshape "tensor.collapse_shape" [2, 3] [5] [[0, 1]]
#guard !reshape "tensor.collapse_shape" [2, 3] [6] [[1, 0]]
#guard !reshape "tensor.expand_shape" [6] [2, 3] [[0], [1]]

end MerlinVerify.Tests.Definedness
