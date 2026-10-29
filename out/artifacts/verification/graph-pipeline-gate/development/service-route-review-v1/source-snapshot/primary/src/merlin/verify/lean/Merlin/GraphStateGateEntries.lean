/- Minimal complete-entry adapter for the unchanged scalar semantics.
The Entry fields, evalEntry, PreservesEntries and roster proofs are retained
from GraphStatePipelineProgram under a new namespace. Only unused dependency
imports and unused Entry typeclass derivations are omitted. Every body still
uses Scalar.Program / Scalar.Model / Scalar.evalProgram directly. Missing
entries return none; bad argument arity returns the same .arity error and world.
No loop/index/resource semantics or new body evaluator is introduced.
This additive source candidate is UNCOMPILED pending exact context review. -/
import Merlin.GraphStateScalar

namespace Merlin.GraphState.GateEntries
set_option autoImplicit false
open Scalar

structure Entry where
  inputs : List Ty
  outputs : List Ty
  body : Program

def evalEntry {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (entries : List Entry)
    (identity : Nat) (arguments : List Value) (world : World) : Option (Outcome Value World Error) :=
  match entries[identity]? with
  | none => none
  | some entry => some (if arguments.length = entry.inputs.length then
      evalProgram model entry.body arguments world else .error (.arity,world))

def PreservesEntries {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (before after : List Entry) : Prop :=
  ∀ identity arguments world, evalEntry model before identity arguments world =
    evalEntry model after identity arguments world

theorem entries_nil {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) : PreservesEntries model [] [] := by
  intro identity arguments world
  rfl

/-- Every defined entry uses its own checked signature/arity; arbitrary external
observers are in the complete outcome. No private environment equality is used. -/
theorem entries_cons {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) {first second : Entry} {before after : List Entry}
    (inputs : first.inputs = second.inputs) (_outputs : first.outputs = second.outputs)
    (body : ∀ arguments world, arguments.length = first.inputs.length →
      evalProgram model first.body arguments world = evalProgram model second.body arguments world)
    (rest : PreservesEntries model before after) :
    PreservesEntries model (first::before) (second::after) := by
  intro identity arguments world
  cases identity with
  | zero =>
    simp only [evalEntry,List.getElem?_cons_zero]
    rw [← inputs]
    by_cases arity : arguments.length = first.inputs.length
    · simp only [arity,↓reduceIte,body arguments world arity]
    · simp only [arity,↓reduceIte]
  | succ identity =>
    simpa only [evalEntry,List.getElem?_cons_succ] using rest identity arguments world

def checkEntries (registry : List Signature) (entries : List Entry) : Bool :=
  entries.all (fun entry => checkTypes registry entry.inputs entry.outputs entry.body)

theorem checkEntries_sound {registry : List Signature} {entries : List Entry}
    (accepted : checkEntries registry entries = true) :
    ∀ entry ∈ entries, Typed registry entry.inputs entry.outputs entry.body := by
  intro entry member
  have checked := List.all_eq_true.mp accepted entry member
  exact checkTypes_sound _ _ _ checked

end Merlin.GraphState.GateEntries
