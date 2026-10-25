/- Exact row-major values across the collapse/expand pair used before broadcast.
Shape/count and coordinate premises are explicit; poison bits are preserved. -/
import MerlinVerify.Rules.ReshapeIndex

namespace MerlinVerify.Rules.W8A8RowReshape

open MerlinVerify Check

theorem append_inBounds {leftShape left rightShape right : List Nat}
    (leftBounds : Idx.InBounds leftShape left) (rightBounds : Idx.InBounds rightShape right) :
    Idx.InBounds (leftShape ++ rightShape) (left ++ right) := by
  induction leftShape generalizing left with
  | nil =>
    have empty : left = [] := (Idx.inBounds_nil_iff _).mp leftBounds
    subst left
    exact rightBounds
  | cons extent shape induction =>
    cases left with
    | nil => simp [Idx.inBounds_cons_nil] at leftBounds
    | cons index rest =>
      have bounded := (Idx.inBounds_cons_cons ..).mp leftBounds
      exact (Idx.inBounds_cons_cons ..).mpr ⟨bounded.1, induction bounded.2⟩

theorem linear_append (leftShape rightShape left right : List Nat) (lengths : leftShape.length = left.length) :
    linearIndex (leftShape ++ rightShape) (left ++ right) =
      linearIndex leftShape left * shapeSize rightShape + linearIndex rightShape right := by
  induction leftShape generalizing left with
  | nil =>
    have empty : left = [] := List.length_eq_zero_iff.mp lengths.symm
    subst left
    simp [linearIndex]
  | cons extent shape induction =>
    cases left with
    | nil => simp at lengths
    | cons index rest =>
      have remaining : shape.length = rest.length := by simpa using lengths
      simp only [List.cons_append, linearIndex_cons, shapeSize_append, induction _ remaining,
        Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]

theorem linear_append_unit (shape row : List Nat) (bounds : Idx.InBounds shape row) :
    linearIndex (shape ++ [1]) (row ++ [0]) = linearIndex shape row := by
  rw [linear_append shape [1] row [0] bounds.1.symm]
  simp [shapeSize, linearIndex]

theorem append_unit_get (tensor : Tensor) (row : Idx) (bounds : Idx.InBounds tensor.shape row) :
    (ReshapeIndex.reshape tensor (tensor.shape ++ [1]) tensor.elt).get (row ++ [0]) = tensor.get row := by
  have extended : Idx.InBounds (tensor.shape ++ [1]) (row ++ [0]) := append_inBounds bounds (by decide)
  simp only [ReshapeIndex.reshape, Tensor.canon, extended, ↓reduceIte,
    linear_append_unit tensor.shape row bounds, ReshapeIndex.delinearize_linear tensor.shape row bounds]

/-- The actual intermediate flat shape is retained in the premise; a matching
element count is checked rather than assuming a particular captured row count. -/
theorem collapsed_then_unit_get (tensor : Tensor) (flat : List Nat) (row : Idx)
    (size : shapeSize flat = shapeSize tensor.shape) (bounds : Idx.InBounds tensor.shape row) :
    (ReshapeIndex.reshape (ReshapeIndex.reshape tensor flat tensor.elt)
      (tensor.shape ++ [1]) tensor.elt).get (row ++ [0]) = tensor.get row := by
  have equalSize : shapeSize (tensor.shape ++ [1]) = shapeSize flat := by
    rw [size, shapeSize_append]
    simp [shapeSize]
  rw [ReshapeIndex.compose tensor flat (tensor.shape ++ [1]) tensor.elt equalSize]
  exact append_unit_get tensor row bounds

end MerlinVerify.Rules.W8A8RowReshape
