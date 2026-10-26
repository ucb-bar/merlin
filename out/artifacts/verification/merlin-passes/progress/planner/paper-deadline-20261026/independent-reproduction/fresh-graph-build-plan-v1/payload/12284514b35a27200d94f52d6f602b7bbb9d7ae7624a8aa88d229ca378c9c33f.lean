/- Actual final SCF control operations, an additive candidate interpretation.
UNCOMPILED. Integer index arithmetic below models only fitting nonnegative
controls; scalar/tensor arithmetic remains an arbitrary ordered partial relation.
No native bit-level fidelity, allocator guarantee or whole-run equality is an
axiom or conclusion. The old graph/epoch semantics are unchanged. -/
import Merlin.GraphLoopsShapeEpochMeaning

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

inductive IndexOpcode where
  | add
  | minimum
  deriving DecidableEq, Repr

def IndexOpcode.apply (opcode : IndexOpcode) (left right : Nat) : Nat :=
  match opcode with
  | .add => left + right
  | .minimum => min left right

/-- The bounded interpretation exposes both input and result ranges. On this
domain unsigned comparison agrees with ordering of nonnegative index values;
connecting it to an implementation of MLIR index arithmetic is still fidelity. -/
def IndexAction {Value World : Type} (bits : Nat) (opcode : IndexOpcode)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) : Prop :=
  match inputs with
  | [.index left, .index right] =>
      GraphLoopsSegment.FitsSigned bits left ∧ GraphLoopsSegment.FitsSigned bits right ∧
      GraphLoopsSegment.FitsSigned bits (opcode.apply left right) ∧
      outputs = [.index (opcode.apply left right)] ∧ after = before
  | _ => False

abbrev IndexOpcodes := List (String × IndexOpcode)

def indexOpcodeAt : IndexOpcodes → String → Option IndexOpcode
  | [], _ => none
  | (key, opcode) :: rest, actual => if actual = key then some opcode else indexOpcodeAt rest actual

def withIndex {Value World : Type} (base : Interpretation Value World) (bits : Nat)
    (opcodes : IndexOpcodes) : Interpretation Value World where
  action := fun key inputs before outputs after => match indexOpcodeAt opcodes key with
    | some opcode => IndexAction bits opcode inputs before outputs after
    | none => base.action key inputs before outputs after
  region := base.region

/-- The same existing load/store, allocation/free, scalar and SCF definitions
are reused. Only the actual generated index-add/min keys receive new cases. -/
def finalMeaning {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (scalar : Epoch.BinaryMeaning Value) :
    Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
  withMemory (withIndex (Epoch.scalarAllocationBase opcodes scalar) bits indices) (Epoch.memoryOpcode opcodes)

theorem generated_action {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (scalar : Epoch.BinaryMeaning Value) (key : String) (opcode : IndexOpcode)
    (selected : indexOpcodeAt indices key = some opcode)
    (notMemory : Epoch.memoryOpcode opcodes key = .other) :
    (finalMeaning (User := User) bits indices opcodes scalar).action key = IndexAction bits opcode := by
  simp only [finalMeaning, withMemory, notMemory, withIndex, selected]

theorem unchanged_action {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (scalar : Epoch.BinaryMeaning Value) (key : String)
    (unchanged : indexOpcodeAt indices key = none) :
    (finalMeaning (User := User) bits indices opcodes scalar).action key =
      (Epoch.graphMeaning opcodes scalar).action key := by
  unfold finalMeaning Epoch.graphMeaning
  cases selected : Epoch.memoryOpcode opcodes key <;>
    simp only [withMemory, selected, withIndex, unchanged]

theorem unchanged_regions {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (scalar : Epoch.BinaryMeaning Value) :
    (finalMeaning (User := User) bits indices opcodes scalar).region =
      (Epoch.graphMeaning opcodes scalar).region := rfl

/-- Existing signed strip-mining bounds justify the two ACTUAL generated
control instructions for every live outer IV, without trip-count unrolling. -/
theorem generated_controls {Value World : Type} (bounds : GraphLoopsSegment.Bounds)
    (valid : bounds.Valid) (outer : Nat) (live : outer < bounds.upper) (world : World) :
    IndexAction (Value := Value) bounds.bits .add [.index outer, .index (bounds.step * bounds.tile)] world
      [.index (outer + bounds.step * bounds.tile)] world ∧
    IndexAction (Value := Value) bounds.bits .minimum
      [.index (outer + bounds.step * bounds.tile), .index bounds.upper] world
      [.index (min (outer + bounds.step * bounds.tile) bounds.upper)] world := by
  have constants := GraphLoopsSegment.Bounds.constants valid
  have controls := GraphLoopsSegment.Bounds.outer_control valid live
  exact ⟨⟨controls.1, constants.2.2.2, controls.2.1, rfl, rfl⟩,
    ⟨controls.2.1, constants.2.1, controls.2.2, rfl, rfl⟩⟩

end Merlin.GraphLoopsMemorySCF.After
