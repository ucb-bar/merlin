/-
G0-FROZEN. Runtime values.

Scalars are bit vectors (integers, `index` at the claim's index width) or float bit patterns.
`poison` is per element. Tensors are functional: `get` is consulted only at in-bounds indices of
`shape`, so two tensors are compared extensionally over their index space (see `Refine.lean`).
-/
import MerlinVerify.Core.AST

namespace MerlinVerify

inductive Scalar where
  /-- An integer or `index` value of the given width. -/
  | bits (w : Nat) (v : BitVec w)
  | flt (k : FKind) (v : BitVec k.width)
  deriving DecidableEq, Repr

inductive Val where
  | poison
  | some (s : Scalar)
  deriving DecidableEq, Repr

/-- A multi-index into a static shape: one coordinate per dimension. -/
abbrev Idx := List Nat

def Idx.InBounds (shape : List Nat) (i : Idx) : Prop :=
  i.length = shape.length ∧ ∀ (d : Nat) (h₁ : d < i.length) (h₂ : d < shape.length), i[d] < shape[d]

instance (shape : List Nat) (i : Idx) : Decidable (Idx.InBounds shape i) := by
  unfold Idx.InBounds; exact inferInstance

structure Tensor where
  shape : List Nat
  /-- Element type (a scalar `Ty`). -/
  elt : Ty
  get : Idx → Val

/-- A runtime value bound to an SSA name. -/
inductive RVal where
  | scalar (v : Val)
  | tensor (t : Tensor)

end MerlinVerify
