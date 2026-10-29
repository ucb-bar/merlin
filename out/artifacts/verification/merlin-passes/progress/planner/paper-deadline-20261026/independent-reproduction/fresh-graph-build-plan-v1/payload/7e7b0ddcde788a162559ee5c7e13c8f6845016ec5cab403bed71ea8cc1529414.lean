/- Concrete typed descriptor accesses correspond to old logical-epoch accesses.
All coordinates, loaded/stored values, private registers and complete caller
worlds are related. Bounds follow from admitted shapes; partial reads stay partial. -/
import Merlin.GraphLoopsShapeEpochPrivate
import Merlin.GraphLoopsShapeEpochState

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem clock_private_store {Value User : Type} (bound next object index : Nat) (value : Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) (inside : object < bound) :
    clockWorld bound next ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object index value⟩ =
      ⟨(clockWorld bound next before).observed,
        GraphLoopsReuse.replaceCell (clockWorld bound next before).heap object index value⟩ :=
  private_store_world bound object index value (atClock next before) inside

theorem load_atom {Value User : Type} (bound next : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key operation : String) (access : Access) (result : Binding)
    (selected : specAt specifications key = some (.load access result))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape) (inside : access.object < bound)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (load : meaning.action operation = Load)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (pinned : before.observed.registers access.buffer.name = view access.object access.shape access.element)
    (coordinates : List Nat)
    (indices : readArguments false [] before.observed.registers access.indices = coordinates.map Runtime.index)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action operation (access.buffer :: access.indices) [result]).step meaning before.observed.registers
      (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
        (.load access.object key) before after ∧ env = after.observed.registers ∧ world = clockWorld bound next after := by
  have arguments : readArguments false [] before.observed.registers (access.buffer :: access.indices) =
      view access.object access.shape access.element :: coordinates.map Runtime.index := by
    change before.observed.registers access.buffer.name :: readArguments false [] before.observed.registers access.indices = _
    rw [pinned, indices]
  simp only [Atom.step, load, arguments, load_iff,
    load_step_iff bound specifications scalar mapping key access result selected objects configs shape]
  constructor
  · rintro ⟨values, ⟨value, coordinate, read, rfl, rfl⟩, typed, env⟩
    have privateRead : before.heap access.object (GraphLoopsReuse.Geometry.flatten access.shape coordinates) = some value := by
      exact (read_private bound access.object _ before.heap before.observed.callerHeap inside).symm.trans read
    have holds : result.Holds (.scalar access.element value) := by cases typed with | cons holds tail => exact holds
    refine ⟨⟨writeRegister before.observed result.name (.scalar access.element value), before.heap⟩,
      ⟨coordinates, value, ⟨pinned, indices, coordinate⟩, privateRead, holds, rfl⟩, env, rfl⟩
  · rintro ⟨after, ⟨actual, value, ⟨bufferRead, indicesRead, coordinate⟩, read, holds, rfl⟩, rfl, rfl⟩
    have same : actual = coordinates := indices_injective actual coordinates (indicesRead.symm.trans indices)
    subst actual
    refine ⟨[.scalar access.element value], ⟨value, coordinate, ?_, rfl, rfl⟩, .cons holds .nil, rfl⟩
    exact (read_private bound access.object _ before.heap before.observed.callerHeap inside).trans read

theorem store_atom {Value User : Type} (bound next : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key operation : String) (access : Access) (operand : Binding)
    (selected : specAt specifications key = some (.store access operand))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape) (inside : access.object < bound)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (store : meaning.action operation = Store)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (pinned : before.observed.registers access.buffer.name = view access.object access.shape access.element)
    (coordinates : List Nat)
    (indices : readArguments false [] before.observed.registers access.indices = coordinates.map Runtime.index)
    (value : Value) (scalarRead : before.observed.registers operand.name = .scalar access.element value)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action operation (operand :: access.buffer :: access.indices) []).step meaning before.observed.registers
      (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
        (.store access.object key) before after ∧ env = after.observed.registers ∧ world = clockWorld bound next after := by
  have arguments : readArguments false [] before.observed.registers (operand :: access.buffer :: access.indices) =
      .scalar access.element value :: view access.object access.shape access.element :: coordinates.map Runtime.index := by
    change before.observed.registers operand.name :: before.observed.registers access.buffer.name ::
      readArguments false [] before.observed.registers access.indices = _
    rw [scalarRead, pinned, indices]
  simp only [Atom.step, store, arguments, store_iff,
    store_step_iff bound specifications scalar mapping key access operand selected objects configs shape]
  constructor
  · rintro ⟨values, ⟨coordinate, rfl, world⟩, _, env⟩
    refine ⟨⟨before.observed, GraphLoopsReuse.replaceCell before.heap access.object
      (GraphLoopsReuse.Geometry.flatten access.shape coordinates) value⟩,
      ⟨coordinates, value, ⟨pinned, indices, coordinate⟩, scalarRead, rfl⟩, env, ?_⟩
    exact world.trans (clock_private_store bound next access.object _ value before inside).symm
  · rintro ⟨after, ⟨actual, stored, ⟨bufferRead, indicesRead, coordinate⟩, storedRead, rfl⟩, rfl, rfl⟩
    have coordinatesSame : actual = coordinates := indices_injective actual coordinates (indicesRead.symm.trans indices)
    have valueSame : stored = value := (Runtime.scalar.inj (storedRead.symm.trans scalarRead)).2
    subst actual
    subst stored
    exact ⟨[], ⟨coordinate, rfl, clock_private_store bound next access.object _ value before inside⟩, .nil, rfl⟩

end Merlin.GraphLoopsShape.Epoch
