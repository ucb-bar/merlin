/- Complete post-allocation effect declarations and independent scalar buffers.
Serialized facts are checked data; their physical truth remains an instruction
contract. The qualified v1 source leaves remain unchanged. -/
import Merlin.GraphRegionsDeclaredBindings

namespace Merlin.GraphRegionsDeclared

structure DeclaredAccess where
  window : Window
  writes : Bool
  deriving DecidableEq, Repr

structure CompleteEffect where
  role : String
  accesses : List DeclaredAccess
  completion : String
  token : Option String
  waits : List String
  exactCopy : Bool
  elements : Nat
  byteOrders : List String
  stages : List String
  readout : Option String
  scalarEncoding : String
  scalarLength : Nat
  scalarBytes : List UInt8
  deriving DecidableEq, Repr

def effectFrameValid (bound : BoundCall) (effect : CompleteEffect) : Prop :=
  effect.role = bound.role ∧ effect.accesses.map (·.window) = bound.reads ++ [bound.output] ∧
  effect.accesses.map (·.writes) = List.replicate bound.reads.length false ++ [true] ∧
  effect.completion = bound.completion ∧ effect.token = bound.token ∧ effect.waits = bound.waits ∧
  effect.exactCopy = false

def effectScalarValid (elements : Nat) (bound : BoundCall) (effect : CompleteEffect) (actualBytes : List UInt8) : Prop :=
  effect.elements = elements ∧ effect.byteOrders = bound.byteOrders ∧ effect.stages = [bound.instruction] ∧
  effect.readout = none ∧ effect.scalarEncoding = "hex" ∧ effect.scalarLength = effect.scalarBytes.length ∧
  effect.scalarBytes = actualBytes

instance (bound : BoundCall) (effect : CompleteEffect) : Decidable (effectFrameValid bound effect) := by
  unfold effectFrameValid; infer_instance

instance (elements : Nat) (bound : BoundCall) (effect : CompleteEffect) (actualBytes : List UInt8) :
    Decidable (effectScalarValid elements bound effect actualBytes) := by unfold effectScalarValid; infer_instance

def checkCompleteEffect (elements : Nat) (bound : BoundCall) (effect : CompleteEffect) (actualBytes : List UInt8) : Bool :=
  decide (effectFrameValid bound effect ∧ effectScalarValid elements bound effect actualBytes)

theorem checkCompleteEffect_iff {elements : Nat} {bound : BoundCall} {effect : CompleteEffect}
    {actualBytes : List UInt8} :
    checkCompleteEffect elements bound effect actualBytes = true ↔
      effectFrameValid bound effect ∧ effectScalarValid elements bound effect actualBytes := by
  simp only [checkCompleteEffect, decide_eq_true_eq]

/-- Acceptance preserves exact read/write order and binds the actual post-pass
scalar bytes. A separately authoritative export/match supplies their scalar AST
meaning. Native truth of this declaration is NOT inferred. -/
theorem checkCompleteEffect_sound {elements : Nat} {bound : BoundCall} {effect : CompleteEffect} {actualBytes : List UInt8}
    (accepted : checkCompleteEffect elements bound effect actualBytes = true) :
    effect.accesses.map (·.window) = bound.reads ++ [bound.output] ∧
      effect.accesses.map (·.writes) = List.replicate bound.reads.length false ++ [true] ∧
      effect.scalarBytes = actualBytes := by
  have valid : effectFrameValid bound effect ∧ effectScalarValid elements bound effect actualBytes :=
    of_decide_eq_true accepted
  exact ⟨valid.1.2.1, valid.1.2.2.1, valid.2.2.2.2.2.2.2⟩

end Merlin.GraphRegionsDeclared
