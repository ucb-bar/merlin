/- Regression statements for the distinction between support, refinement, and definedness. -/
import MerlinVerify.Sem.Defined
import MerlinVerify.Check.Typing
import MerlinVerify.Check.DefinedLinalg
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

def dense (splat : Bool) (bytes : List UInt8) (shape : List Nat := [2]) : Op :=
  mkOp "arith.constant" [v 0 (tensorTy shape i32)] []
    [("value", .dense (tensorTy shape i32) splat bytes)]

#guard accepts [] [dense true [7, 0, 0, 0], ret [0]] [] [.tensor [2] i32]
#guard accepts [] [dense false [1, 0, 0, 0, 2, 0, 0, 0], ret [0]] [] [.tensor [2] i32]
#guard accepts [] [dense false [] [0], ret [0]] [] [.tensor [0] i32]
#guard !accepts [] [dense true [7, 0, 0], ret [0]] [] [.tensor [2] i32]
#guard !accepts [] [dense false [1, 0, 0, 0], ret [0]] [] [.tensor [2] i32]
#guard !accepts [] [dense false [1, 0, 0, 0, 2, 0, 0, 0, 0], ret [0]] [] [.tensor [2] i32]
#guard !accepts [] [mkOp "arith.constant" [v 0 (tensorTy [3] i32)] []
  [("value", .dense (tensorTy [2] i32) true [7, 0, 0, 0])], ret [0]] [] [.tensor [3] i32]

#guard readInts (Sem.Ops.denote ctx { ops := [func "f" [] [dense true [7, 0, 0, 0], ret [0]]] } "f" [])
  == some [[some 7, some 7]]
#guard readInts (Sem.Ops.denote ctx { ops := [func "f" []
  [dense false [255, 255, 255, 255, 2, 0, 0, 0], ret [0]]] } "f" []) == some [[some (-1), some 2]]

end MerlinVerify.Tests.Definedness

namespace MerlinVerify.Tests.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Tests.Semantics

def cast (name : String) (src dst : Nat) (attrs : List (String × Attr) := []) : Bool :=
  accepts [v 0 (.int src)] [mkOp name [v 1 (.int dst)] [0] attrs, ret [1]]
    [.scalar (.int src)] [.scalar (.int dst)]

#guard cast "arith.extsi" 8 32
#guard cast "arith.extui" 8 32
#guard cast "arith.trunci" 32 8
#guard !cast "arith.extsi" 32 8
#guard !cast "arith.extui" 8 8
#guard !cast "arith.trunci" 8 32
#guard !cast "arith.extui" 8 32 [("nonNeg", .unit)]
#guard !cast "arith.trunci" 32 8 [("overflowFlags", .enum "arith" "overflow" "nsw")]
#guard !cast "arith.trunci" 32 8 [("overflowFlags", .enum "arith" "overflow" "nuw")]

end MerlinVerify.Tests.Definedness

namespace MerlinVerify.Tests.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Tests.Semantics

def capturedBody : Region := .mk [.mk [v 1 i32]
  [mkOp "arith.addi" [v 2 i32] [0, 1], mkOp "linalg.yield" [] [2]]]

#guard checkRegion (book 64) [(0, ⟨.scalar i32, true⟩)] capturedBody
  [⟨.scalar i32, true⟩] [.scalar i32]
#guard !checkRegion (book 64) [] capturedBody [⟨.scalar i32, true⟩] [.scalar i32]
#guard !checkRegion (book 64) [(0, ⟨.scalar i32, false⟩)] capturedBody
  [⟨.scalar i32, true⟩] [.scalar i32]

-- The evaluator and analysis must both give block arguments precedence over captures.
def shadowedBody : Region := .mk [.mk [v 0 i32] [mkOp "linalg.yield" [] [0]]]

#guard checkRegion (book 64) [(0, ⟨.scalar i32, false⟩)] shadowedBody
  [⟨.scalar i32, true⟩] [.scalar i32]
#guard !checkRegion (book 64) [(0, ⟨.scalar i32, true⟩)] shadowedBody
  [⟨.scalar i32, false⟩] [.scalar i32]
#guard !checkRegion (book 64) [] shadowedBody [⟨.scalar i8, true⟩] [.scalar i32]
#guard !checkRegion (book 64) [] (.mk [.mk [] [], .mk [] []]) [] []

end MerlinVerify.Tests.Definedness

namespace MerlinVerify.Tests.Definedness

open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Tests.Semantics

-- An actual integer tensor contraction, including widening casts in its reduction body.
#guard check (book 64) matmulModule "mm" [.tensor [2, 3] i8, .tensor [3, 2] i8] [.tensor [2, 2] i32]

def fill (sh : List Nat) (body : Region) : Op :=
  mkOp "linalg.fill" [v 2 (tensorTy sh i32)] [0, 1]
    [("operandSegmentSizes", .denseArray i32 [1, 1])] [] [body]

def fillBody (yieldId : Nat) : Region :=
  .mk [.mk [v 10 i32, v 11 i32] [mkOp "linalg.yield" [] [yieldId]]]

def fills (sh : List Nat) (body : Region) : Bool :=
  accepts [v 0 i32] [mkOp "tensor.empty" [v 1 (tensorTy sh i32)] [], fill sh body, ret [2]]
    [.scalar i32] [.tensor sh i32]

#guard fills [2, 3] (fillBody 10)
#guard fills [] (fillBody 10)
#guard fills [0, 3] (fillBody 10)
-- Reading the uninitialized destination cannot certify a fill.
#guard !fills [2, 3] (fillBody 11)
-- Captures use their actual enclosing facts.
#guard fills [2, 3] (fillBody 0)
#guard !fills [2, 3] (fillBody 99)
#guard !fills [2, 3] (.mk [.mk [v 10 i32, v 11 i32]
  [mkOp "arith.addi" [v 12 i32] [10, 11], mkOp "linalg.yield" [] [12]]])
#guard fills [2, 3] (.mk [.mk [v 10 i32, v 11 i32]
  [mkOp "arith.addi" [v 12 i32] [10, 0], mkOp "linalg.yield" [] [12]]])
#guard !fills [2, 3] (.mk [.mk [v 10 i32] [mkOp "linalg.yield" [] [10]]])
#guard !fills [2, 3] (.mk [.mk [v 10 i32, v 11 i32] [mkOp "linalg.yield" [] [10, 10]]])

end MerlinVerify.Tests.Definedness
