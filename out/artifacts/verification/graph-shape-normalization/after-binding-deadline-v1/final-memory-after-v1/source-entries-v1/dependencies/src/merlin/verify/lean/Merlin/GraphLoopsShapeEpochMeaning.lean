/- One concrete graph interpretation selected from actual operation metadata.
Unchanged binary arithmetic is partial and ordered. Memory/SCF are the existing
explicit operations; private allocation uses a disclosed logical naming clock.
Unknown operations refuse. Native operation/allocator fidelity remains separate. -/
import Merlin.GraphLoopsShapeEpochPrimitives

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

inductive Opcode where
  | load
  | store
  | loop
  | binary (element : String)
  | allocate (shape : List Nat) (element : String)
  | release
  deriving DecidableEq, Repr

abbrev Opcodes := List (String × Opcode)

def opcodeAt : Opcodes → String → Option Opcode
  | [], _ => none
  | (key, opcode) :: rest, actual => if actual = key then some opcode else opcodeAt rest actual

def memoryOpcode (opcodes : Opcodes) (key : String) : MemoryOpcode :=
  match opcodeAt opcodes key with
  | some .load => .load
  | some .store => .store
  | some .loop => .loop
  | _ => .other

def scalarAllocationBase {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) :
    Interpretation Value (MemoryWorld Value (AllocationPublic User)) where
  action := fun key inputs before outputs after => match opcodeAt opcodes key with
    | some (.binary element) => Binary element (scalar key) inputs before outputs after
    | some (.allocate shape element) => Allocate shape element inputs before outputs after
    | some .release => Release inputs before outputs after
    | _ => False
  region := fun _ _ _ _ _ _ => False

def graphMeaning {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) :
    Interpretation Value (MemoryWorld Value (AllocationPublic User)) :=
  withMemory (scalarAllocationBase opcodes scalar) (memoryOpcode opcodes)

theorem graph_load {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) (key : String)
    (selected : opcodeAt opcodes key = some .load) :
    (graphMeaning (User := User) opcodes scalar).action key = Load := by
  simp only [graphMeaning, withMemory, memoryOpcode, selected]

theorem graph_store {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) (key : String)
    (selected : opcodeAt opcodes key = some .store) :
    (graphMeaning (User := User) opcodes scalar).action key = Store := by
  simp only [graphMeaning, withMemory, memoryOpcode, selected]

theorem graph_binary {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) (key element : String)
    (selected : opcodeAt opcodes key = some (.binary element)) :
    (graphMeaning (User := User) opcodes scalar).action key = Binary element (scalar key) := by
  simp only [graphMeaning, withMemory, memoryOpcode, scalarAllocationBase, selected]

theorem graph_allocate {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value)
    (key : String) (shape : List Nat) (element : String)
    (selected : opcodeAt opcodes key = some (.allocate shape element)) :
    (graphMeaning (User := User) opcodes scalar).action key = Allocate shape element := by
  simp only [graphMeaning, withMemory, memoryOpcode, scalarAllocationBase, selected]

theorem graph_release {Value User : Type} (opcodes : Opcodes) (scalar : BinaryMeaning Value) (key : String)
    (selected : opcodeAt opcodes key = some .release) :
    (graphMeaning (User := User) opcodes scalar).action key = Release := by
  simp only [graphMeaning, withMemory, memoryOpcode, scalarAllocationBase, selected]

/-- Only dispatch changes the meaning at internal-call keys. Actual non-call
keys retain their concrete operation meaning at every finite call depth. -/
theorem dispatched_action {Value World : Type} (meaning : Interpretation Value World) (module : Module)
    (calls : String → RegionMeaning Value World) (key : String) (notCall : callTarget module.calls key = none) :
    (withCalls meaning module calls).action key = meaning.action key := by
  simp only [withCalls, notCall]

theorem meaning_extensional {Value World : Type} (first second : Interpretation Value World)
    (actions : first.action = second.action) (regions : first.region = second.region) : first = second := by
  cases first
  cases second
  cases actions
  cases regions
  rfl

/-- Commuting dispatch with concrete memory is a proved pointwise identity.
The classifier's load/store keys must be absent from the internal call table. -/
theorem calls_with_memory {Value User : Type} (base : Interpretation Value (MemoryWorld Value User))
    (classify : String → MemoryOpcode) (module : Module)
    (calls : String → RegionMeaning Value (MemoryWorld Value User))
    (nonCall : ∀ key, classify key = .load ∨ classify key = .store → callTarget module.calls key = none) :
    withCalls (withMemory base classify) module calls = withMemory (withCalls base module calls) classify := by
  apply meaning_extensional
  · funext key inputs before outputs after
    cases selected : classify key with
    | load => simp only [withCalls, withMemory, selected, nonCall key (Or.inl selected)]
    | store => simp only [withCalls, withMemory, selected, nonCall key (Or.inr selected)]
    | loop => simp only [withCalls, withMemory, selected]
    | other => simp only [withCalls, withMemory, selected]
  · rfl

end Merlin.GraphLoopsShape.Epoch
