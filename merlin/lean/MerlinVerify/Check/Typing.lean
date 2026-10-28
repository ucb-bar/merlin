/-
Static value types for the progress (“fully modelled”) theorem (design §4.5).

A claim at a boundary is conditional refinement: it holds wherever the source is modelled. The
end-to-end claim is only meaningful if the capture is never `stuck` on any well-typed input, so
this file defines what a well-typed runtime value is; `Check/Progress.lean` proves that a statically
checked program evaluates without getting stuck.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Check

open MerlinVerify

/-- The static type of an SSA value: a scalar type, or a static, unencoded tensor of a scalar type. -/
inductive VTy where
  | scalar (t : Ty)
  | tensor (shape : List Nat) (elt : Ty)
  deriving DecidableEq

/-- Static type of an MLIR type at index width `ib`, if it is one the semantics models. -/
def vtyOf (ib : Nat) : Ty → Option VTy
  | .tensor shape elt Option.none =>
    if (elt.bitWidth? ib).isSome then (shape.mapM id).map (VTy.tensor · elt) else Option.none
  | t => if (t.bitWidth? ib).isSome then some (.scalar t) else Option.none

def Val.HasTy (ib : Nat) (t : Ty) : Val → Prop
  | .poison => True
  | .some s => s.conforms ib t = true

def RVal.HasTy (ib : Nat) : RVal → VTy → Prop
  | .scalar v, .scalar t => Val.HasTy ib t v
  | .tensor x, .tensor sh elt =>
    x.shape = sh ∧ x.elt = elt ∧ ∀ i, Idx.InBounds sh i → Val.HasTy ib elt (x.get i)
  | _, _ => False

/-- Values have the given types, position by position. -/
def HasTys (ib : Nat) (vs : List RVal) (ts : List VTy) : Prop := List.Forall₂ (RVal.HasTy ib) vs ts

/-- Typing environment: the static type of every SSA value in scope (newest first). -/
abbrev TEnv := List (ValueId × VTy)

/-- Every typed value is bound, to a value of its type. -/
def EnvTyped (ib : Nat) (Γ : TEnv) (env : Env) : Prop :=
  ∀ x t, Γ.lookup x = some t → ∃ v, env.get? x = some v ∧ RVal.HasTy ib v t

end MerlinVerify.Check
