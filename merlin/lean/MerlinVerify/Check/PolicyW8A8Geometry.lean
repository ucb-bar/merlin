/- Geometry shared by W8A8 pointwise stages. Expected parameter dimensions must come from
the source contract in the future site rule; candidate output maps cannot establish that choice. -/
import MerlinVerify.Check.StructuredRuntime

namespace MerlinVerify.Check.PolicyW8A8Geometry

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def parameterMap (rank : Nat) (dims : List Nat) : AffineMap :=
  { numDims := rank, numSyms := 0, results := dims.map AffineExpr.dim }

def parameterShape (shape dims : List Nat) : List Nat := dims.map (shape.getD · 0)

def DimensionsValid (shape dims : List Nat) : Prop := dims.Nodup ∧ ∀ d ∈ dims, d < shape.length

instance (shape dims : List Nat) : Decidable (DimensionsValid shape dims) := by
  unfold DimensionsValid
  infer_instance

def checkDimensions (shape dims : List Nat) : Bool := decide (DimensionsValid shape dims)

/-- The semantic layout obligation is stated before the executable matcher. -/
def ParallelLayout (inputs : List VTy) (shape : List Nat) (elt : Ty)
    (expected maps : List AffineMap) (iters : List IterKind) : Prop :=
  maps = expected ++ [identityMap shape.length] ∧
  iters = List.replicate shape.length .parallel ∧
  StructuredOk maps iters.length (inputs ++ [.tensor shape elt]) shape

def checkParallel (inputs : List VTy) (shape : List Nat) (elt : Ty)
    (expected maps : List AffineMap) (iters : List IterKind) : Bool :=
  decide (maps = expected ++ [identityMap shape.length]) &&
  decide (iters = List.replicate shape.length .parallel) &&
  (structuredRanges? maps iters.length (inputs ++ [.tensor shape elt]) == some shape)

theorem checkParallel_sound {inputs : List VTy} {shape : List Nat} {elt : Ty}
    {expected maps : List AffineMap} {iters : List IterKind}
    (h : checkParallel inputs shape elt expected maps iters = true) :
    ParallelLayout inputs shape elt expected maps iters := by
  simp only [checkParallel, Bool.and_eq_true] at h
  exact ⟨of_decide_eq_true h.1.1, of_decide_eq_true h.1.2,
    structuredRanges?_spec (eq_of_beq h.2)⟩

theorem checkDimensions_sound {shape dims : List Nat} (h : checkDimensions shape dims = true) :
    DimensionsValid shape dims := of_decide_eq_true h

end MerlinVerify.Check.PolicyW8A8Geometry
