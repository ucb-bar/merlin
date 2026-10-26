/- Hash-free source/declaration correspondence for the actual narrow scalar DAG.

Soundness target: complete actual scalar AST matching implies identical ordered
evaluation under ANY common admitted primitive interpretation, including exact
argument/result widths, literal properties and SSA return selection. This layer
does not import the unqualified source-operational profile or assume SHA
injectivity. Instruction fidelity belongs to the subsequent per-call contracts.
-/
import MerlinVerify.Core.Eq
import Merlin.GraphRegionsFusion

namespace Merlin.GraphRegionsDeclared
open MerlinVerify

inductive Binary where
  | add | sub | mul
  deriving DecidableEq, Repr

inductive Cast where
  | trunc | signed | unsigned
  deriving DecidableEq, Repr

inductive Primitive where
  | constant (width : Nat) (literal : Int)
  | binary (width : Nat) (kind : Binary)
  | cast (source target : Nat) (kind : Cast)
  deriving DecidableEq, Repr

def Primitive.inputs : Primitive → List Nat
  | .constant .. => []
  | .binary width _ => [width, width]
  | .cast source _ _ => [source]

def Primitive.width : Primitive → Nat
  | .constant width _ | .binary width _ => width
  | .cast _ target _ => target

def Primitive.name : Primitive → String
  | .constant .. => "arith.constant"
  | .binary _ .add => "arith.addi"
  | .binary _ .sub => "arith.subi"
  | .binary _ .mul => "arith.muli"
  | .cast _ _ .trunc => "arith.trunci"
  | .cast _ _ .signed => "arith.extsi"
  | .cast _ _ .unsigned => "arith.extui"

def Primitive.attrs : Primitive → List (String × Attr)
  | .constant width literal => [("value", .int literal (.int width))]
  | .binary .. | .cast _ _ .trunc => [("overflowFlags", .enum "arith" "overflow" "none")]
  | .cast _ _ .signed | .cast _ _ .unsigned => []

def Primitive.Valid : Primitive → Prop
  | .constant width _ | .binary width _ => 0 < width ∧ width ≤ 256
  | .cast source target .trunc => 0 < target ∧ target < source ∧ source ≤ 256
  | .cast source target _ => 0 < source ∧ source < target ∧ target ≤ 256

structure Instruction where
  primitive : Primitive
  operands : List Nat
  deriving DecidableEq, Repr

def Instruction.operation (instruction : Instruction) (result : Nat) : Op :=
  .mk instruction.primitive.name [⟨result, .int instruction.primitive.width⟩]
    instruction.operands instruction.primitive.attrs [] [] []

/-- Ordered canonical SSA slots, not expression fingerprints. Extra unused
operations remain part of this record and cannot be silently discarded. -/
structure Program where
  inputs : List Nat
  instructions : List Instruction
  returned : Nat
  output : Nat
  deriving DecidableEq, Repr

abbrev Environment (Value : Type) := Nat → Value

def bind {α : Type} (environment : Environment α) (id : Nat) (value : α) : Environment α :=
  fun key => if key = id then value else environment key

def run {α : Type} (meaning : Primitive → List α → α) :
    Nat → List Instruction → Environment α → Environment α
  | _, [], environment => environment
  | next, instruction :: rest, environment =>
    run meaning (next + 1) rest
      (bind environment next (meaning instruction.primitive (instruction.operands.map environment)))

def Program.evaluate {α : Type} (program : Program) (meaning : Primitive → List α → α)
    (arguments : Environment α) : α :=
  run meaning program.inputs.length program.instructions arguments program.returned

/-- Exact structural matching has no cryptographic premise and preserves every
operation in its original order for every common primitive interpretation. -/
theorem structural_evaluation {α : Type} {first second : Program}
    (same : first = second) (meaning : Primitive → List α → α)
    (arguments : Environment α) : first.evaluate meaning arguments = second.evaluate meaning arguments := by
  rw [same]

theorem run_congr {α : Type} (meaning : Primitive → List α → α) (next : Nat)
    (instructions : List Instruction) {first second : Environment α} (same : first = second) :
    run meaning next instructions first = run meaning next instructions second := by
  rw [same]

end Merlin.GraphRegionsDeclared
