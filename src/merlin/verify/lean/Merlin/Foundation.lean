/-
Generic integer semantics and transformation obligations. No target facts or
schedule choices appear here. Primitive hardware conformance is a separate
obligation of the selected software specification and RTL evidence.
-/
import LeanMLIR.Framework.Basic
import Std.Tactic.BVDecide
import Lean.Elab.Tactic.Omega
import Init.Data.List.Range

namespace Merlin

inductive IntegerTy where
  | bits (width : Nat)
  deriving DecidableEq

@[reducible] instance : TyDenote IntegerTy where
  toType | .bits width => BitVec width

inductive IntegerOp where
  | add (width : Nat)
  | sub (width : Nat)
  | mul (width : Nat)

abbrev IntegerDialect : Dialect := { Op := IntegerOp, Ty := IntegerTy }

@[reducible] instance : DialectSignature IntegerDialect where
  signature
    | .add width | .sub width | .mul width =>
      { sig := [.bits width, .bits width], regSig := [], returnTypes := [.bits width] }

@[reducible] instance : DialectDenote IntegerDialect where
  denote
    | .add _, .cons a (.cons b .nil), .nil => .cons (BitVec.add a b) .nil
    | .sub _, .cons a (.cons b .nil), .nil => .cons (BitVec.sub a b) .nil
    | .mul _, .cons a (.cons b .nil), .nil => .cons (BitVec.mul a b) .nil

def bvAdd (width : Nat) (a b : BitVec width) : BitVec width :=
  match DialectDenote.denote (d := IntegerDialect) (.add width) (.cons a (.cons b .nil)) .nil with
  | .cons value .nil => value

def bvSub (width : Nat) (a b : BitVec width) : BitVec width :=
  match DialectDenote.denote (d := IntegerDialect) (.sub width) (.cons a (.cons b .nil)) .nil with
  | .cons value .nil => value

def bvMul (width : Nat) (a b : BitVec width) : BitVec width :=
  match DialectDenote.denote (d := IntegerDialect) (.mul width) (.cons a (.cons b .nil)) .nil with
  | .cons value .nil => value

@[simp] theorem bvAdd_eq (width : Nat) (a b : BitVec width) : bvAdd width a b = a + b := rfl
@[simp] theorem bvSub_eq (width : Nat) (a b : BitVec width) : bvSub width a b = a - b := rfl
@[simp] theorem bvMul_eq (width : Nat) (a b : BitVec width) : bvMul width a b = a * b := rfl

/-- Ordered segmentation preserves any fold, including a fold carrying memory state. -/
theorem orderedSegments (chunks : List (List α)) (body : σ → α → σ) (initial : σ) :
    chunks.foldl (fun state chunk => chunk.foldl body state) initial =
      chunks.flatten.foldl body initial := by
  exact List.foldl_flatten.symm

/-- Split a strided iteration sequence without reassociating its state updates. -/
theorem rangeSplit (start first rest step : Nat) :
    List.range' start first step ++ List.range' (start + step * first) rest step =
      List.range' start (first + rest) step := by
  exact List.range'_append

/-- The quotient/remainder coordinate map reconstructs every logical iteration. -/
theorem tileIndex (index tile : Nat) :
    tile * (index / tile) + index % tile = index := by
  exact Nat.div_add_mod index tile

/-- A logical iteration belongs to the valid tail of its tile. -/
theorem tileTail (index extent tile : Nat) (positive : 0 < tile) (inside : index < extent) :
    index % tile < min tile (extent - tile * (index / tile)) := by
  have remainder := Nat.mod_lt index positive
  have reconstruct := tileIndex index tile
  omega

/-- Disjoint live intervals cannot require the same slot simultaneously. -/
theorem disjointLifetime (aStart aEnd bStart bEnd time : Nat)
    (ordered : aEnd < bStart ∨ bEnd < aStart) :
    ¬ (aStart ≤ time ∧ time ≤ aEnd ∧ bStart ≤ time ∧ time ≤ bEnd) := by
  omega

/-- Adjacent independent state transitions may be exchanged under a checked commute relation. -/
theorem commuteTransitions (first second : σ → σ)
    (commute : ∀ state, second (first state) = first (second state)) (state : σ) :
    second (first state) = first (second state) := commute state

end Merlin
