/- Exact pointwise tensor writes. Every in-bounds cell must be covered; arbitrary
unobserved contents are preserved until the frozen canonicalization boundary. -/
import MerlinVerify.Rules.W8A8TensorTrace

namespace MerlinVerify.Rules.W8A8TensorOverwrite

open MerlinVerify W8A8TensorTrace

variable {α : Type}

theorem preserve_cell (values : Idx → α) (points : List Idx) :
    ∀ (initial : Idx → α) (cell : Idx), initial cell = values cell →
      accumulate id (fun point _ => values point) initial points cell = values cell := by
  induction points with
  | nil => intro initial cell initialized; exact initialized
  | cons point rest induction =>
    intro initial cell initialized
    simp only [accumulate, List.foldl_cons]
    apply induction
    simp only [update, id_eq]
    split
    · next same => exact congrArg values same.symm
    · exact initialized

theorem written_cell (values : Idx → α) (points : List Idx) :
    ∀ (initial : Idx → α) (cell : Idx), cell ∈ points →
      accumulate id (fun point _ => values point) initial points cell = values cell := by
  induction points with
  | nil => intro _ _ member; cases member
  | cons point rest induction =>
    intro initial cell member
    simp only [accumulate, List.foldl_cons]
    rcases List.mem_cons.mp member with same | later
    · subst point
      exact preserve_cell values rest _ cell (by simp only [update, id_eq, ↓reduceIte])
    · exact induction _ cell later

theorem canon_written (shape : List Nat) (element : Ty) (values initial : Idx → Val) (points : List Idx)
    (covers : ∀ cell, Idx.InBounds shape cell → cell ∈ points) :
    Tensor.canon shape element (accumulate id (fun point _ => values point) initial points) =
      Tensor.canon shape element values := by
  unfold Tensor.canon
  congr 1
  funext cell
  split
  · next bounds => exact written_cell values points initial cell (covers cell bounds)
  · rfl

end MerlinVerify.Rules.W8A8TensorOverwrite
