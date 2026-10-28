/- Row-major reshape identities, including empty shapes, rank zero, and poison elements. -/
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Rules.ReshapeIndex

open MerlinVerify MerlinVerify.Check

theorem linear_delinearize : ∀ (shape : List Nat) (k : Nat), k < shapeSize shape →
    linearIndex shape (Sem.Ops.Tensor.delinearize shape k) = k
  | [], k, h => by simp [shapeSize] at h; subst k; rfl
  | d :: ds, k, h => by
    have hp : 0 < shapeSize ds := by
      rw [shapeSize_cons] at h
      exact Nat.pos_of_ne_zero fun he => by rw [he] at h; simp at h
    simp only [Sem.Ops.Tensor.delinearize, linearIndex_cons]
    rw [linear_delinearize ds (k % shapeSize ds) (Nat.mod_lt _ hp)]
    exact Nat.div_add_mod' k (shapeSize ds)

theorem delinearize_linear : ∀ (shape : List Nat) (i : Idx), Idx.InBounds shape i →
    Sem.Ops.Tensor.delinearize shape (linearIndex shape i) = i
  | [], i, h => by rw [(Idx.inBounds_nil_iff i).1 h]; rfl
  | _ :: _, [], h => (Idx.inBounds_cons_nil _ _ h).elim
  | d :: ds, x :: xs, h => by
    obtain ⟨hx, hxs⟩ := (Idx.inBounds_cons_cons d ds x xs).1 h
    have hlt := linearIndex_lt ds xs hxs
    have hp : 0 < shapeSize ds := by omega
    simp only [linearIndex_cons, Sem.Ops.Tensor.delinearize]
    rw [Nat.mul_comm x (shapeSize ds), Nat.mul_add_div hp,
      Nat.div_eq_of_lt hlt, Nat.add_zero,
      Nat.mul_add_mod, Nat.mod_eq_of_lt hlt, delinearize_linear ds xs hxs]

def reshape (t : Tensor) (shape : List Nat) (elt : Ty) : Tensor :=
  Tensor.canon shape elt fun i => t.get (Sem.Ops.Tensor.delinearize t.shape (linearIndex shape i))

/-- Intermediate canonicalization has no effect on the row-major elements subsequently read.
No division by zero is assumed: an empty result has no in-bounds indices. -/
theorem compose (t : Tensor) (mid dst : List Nat) (elt : Ty)
    (hsize : shapeSize dst = shapeSize mid) :
    reshape (reshape t mid elt) dst elt = reshape t dst elt := by
  apply Tensor.canon_ext
  intro i hi
  have hk : linearIndex dst i < shapeSize mid := by rw [← hsize]; exact linearIndex_lt _ _ hi
  have hj := delinearize_inBounds mid _ hk
  simp only [reshape, Tensor.canon, hj, ↓reduceIte]
  rw [linear_delinearize mid _ hk]

/-- Cancellation is extensional. The C1 environment relation uses literal equality, so this
theorem alone does not authorize deleting a reshape and aliasing an arbitrary raw input tensor. -/
theorem cancel (t : Tensor) (mid : List Nat) (hsize : shapeSize t.shape = shapeSize mid) :
    reshape (reshape t mid t.elt) t.shape t.elt = t.canonize := by
  rw [compose t mid t.shape t.elt hsize]
  apply Tensor.canon_ext
  intro i hi
  rw [delinearize_linear t.shape i hi]

end MerlinVerify.Rules.ReshapeIndex
