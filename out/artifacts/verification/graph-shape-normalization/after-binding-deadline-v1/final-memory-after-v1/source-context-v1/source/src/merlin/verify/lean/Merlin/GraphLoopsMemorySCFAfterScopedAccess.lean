/- UNCOMPILED source candidate. Access correspondence uses only live operands.
Extra target registers are unrestricted. Both directions preserve partial reads,
actual coordinates, physical slots and the complete caller/allocator world.
-/
import Merlin.GraphLoopsMemorySCFAfterMemorySteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem RegistersRelated.indices {Value : Type} {objects : GraphLoopsReuse.Objects}
    {context : Context} {logical actual : GraphLoopsShape.Env Value}
    (related : RegistersRelated objects context logical actual) (typed : Typed context logical)
    (indices : List Binding) (bound : AllBound context indices)
    (kinds : ∀ binding ∈ indices, binding.kind = .index) :
    readArguments false [] actual indices = readArguments false [] logical indices := by
  apply related.returns typed indices bound
  intro binding member
  rw [kinds binding member]
  trivial

/-- The load's logical successor and actual SSA definition are derived from the
same initialized physical cell. The exact assignment is retained in addition to
the scoped invariant, so the reverse implication cannot invent scratch values.
-/
theorem scoped_load_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (specifications : Epoch.Specifications)
    (mapping : Aliases) (scalar : Epoch.BinaryMeaning Value)
    (context : Context) (clock : Nat) (key operation : String) (access : Epoch.Access) (result : Binding)
    (selected : Epoch.specAt specifications key = some (.load access result))
    (inside : access.object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects access.object < objects.length)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (bufferBound : access.buffer ∈ context) (indicesBound : AllBound context access.indices)
    (indexKinds : ∀ binding ∈ access.indices, binding.kind = .index)
    (fresh : result.name ∉ names context)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (load : meaning.action operation = Load)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (pinned : before.observed.registers access.buffer.name = Epoch.view access.object access.shape access.element)
    (typed : Typed context before.observed.registers) (actual : GraphLoopsShape.Env Value)
    (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation (access.buffer :: access.indices) [result]).step meaning actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after value,
        GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation objects.length specifications mapping scalar)
          true true (.load access.object key) before after ∧
        after.observed.registers = GraphLoopsShape.put before.observed.registers result.name (.scalar access.element value) ∧
        next = GraphLoopsShape.put actual result.name (.scalar access.element value) ∧
        world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects (result :: context) after.observed.registers next := by
  obtain ⟨coordinates, indices⟩ := Epoch.indices_of_typed context before.observed.registers typed
    access.indices indicesBound indexKinds
  have actualIndices := (related.indices typed access.indices indicesBound indexKinds).trans indices
  have actualBuffer := related.private_view access.buffer bufferBound access.object inside access.shape access.element pinned
  rw [actual_load_iff objects objects.length clock access.object slotInside operation access.buffer result
    access.indices access.shape coordinates access.element meaning load before actual actualBuffer actualIndices]
  constructor
  · rintro ⟨value, coordinate, read, holds, rfl, rfl⟩
    let after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User) :=
      ⟨Epoch.writeRegister before.observed result.name (.scalar access.element value), before.heap⟩
    refine ⟨after, value, ?_, rfl, rfl, rfl, ?_⟩
    · exact (candidate_load_iff objects.length specifications scalar mapping key access result selected objects configs
        shape before after).mpr ⟨coordinates, value, ⟨pinned, indices, coordinate⟩, read, holds, rfl⟩
    · exact related.define result (.scalar access.element value) fresh
  · rintro ⟨after, value, run, assigned, rfl, rfl, _⟩
    obtain ⟨actualCoordinates, loaded, reads, read, holds, rfl⟩ :=
      (candidate_load_iff objects.length specifications scalar mapping key access result selected objects configs
        shape before after).mp run
    have coordinatesEqual := Epoch.indices_injective actualCoordinates coordinates (reads.2.1.symm.trans indices)
    subst actualCoordinates
    have valueEqual : loaded = value := by
      have atResult := congrFun assigned result.name
      simp only [Epoch.writeRegister, GraphLoopsShape.put, ↓reduceIte] at atResult
      exact (Runtime.scalar.inj atResult).2
    subst loaded
    exact ⟨value, reads.2.2, read, holds, rfl, rfl⟩

theorem scoped_store_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (specifications : Epoch.Specifications)
    (mapping : Aliases) (scalar : Epoch.BinaryMeaning Value)
    (context : Context) (clock : Nat) (key operation : String) (access : Epoch.Access) (operand : Binding)
    (selected : Epoch.specAt specifications key = some (.store access operand))
    (inside : access.object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects access.object < objects.length)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (bufferBound : access.buffer ∈ context) (operandBound : operand ∈ context)
    (indicesBound : AllBound context access.indices)
    (indexKinds : ∀ binding ∈ access.indices, binding.kind = .index)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (store : meaning.action operation = Store)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (pinned : before.observed.registers access.buffer.name = Epoch.view access.object access.shape access.element)
    (value : Value) (operandRead : before.observed.registers operand.name = .scalar access.element value)
    (typed : Typed context before.observed.registers) (actual : GraphLoopsShape.Env Value)
    (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation (operand :: access.buffer :: access.indices) []).step meaning actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after,
        GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation objects.length specifications mapping scalar)
          true true (.store access.object key) before after ∧
        next = actual ∧ world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  obtain ⟨coordinates, indices⟩ := Epoch.indices_of_typed context before.observed.registers typed
    access.indices indicesBound indexKinds
  have actualIndices := (related.indices typed access.indices indicesBound indexKinds).trans indices
  have actualBuffer := related.private_view access.buffer bufferBound access.object inside access.shape access.element pinned
  have actualOperand : actual operand.name = .scalar access.element value := by
    rw [related operand operandBound, operandRead]
    rfl
  rw [actual_store_iff objects objects.length clock access.object slotInside operation operand access.buffer
    access.indices access.shape coordinates access.element value meaning store before actual
    actualOperand actualBuffer actualIndices]
  constructor
  · rintro ⟨coordinate, rfl, rfl⟩
    refine ⟨⟨before.observed, GraphLoopsReuse.replaceCell before.heap (GraphLoopsReuse.slotOf objects access.object)
      (GraphLoopsReuse.Geometry.flatten access.shape coordinates) value⟩, ?_, rfl, rfl, related⟩
    exact (candidate_store_iff objects.length specifications scalar mapping key access operand selected objects configs
      shape before _).mpr ⟨coordinates, value, ⟨pinned, indices, coordinate⟩, operandRead, rfl⟩
  · rintro ⟨after, run, rfl, rfl, _⟩
    obtain ⟨actualCoordinates, stored, reads, scalarRead, rfl⟩ :=
      (candidate_store_iff objects.length specifications scalar mapping key access operand selected objects configs
        shape before after).mp run
    have coordinatesEqual := Epoch.indices_injective actualCoordinates coordinates (reads.2.1.symm.trans indices)
    have valueEqual := (Runtime.scalar.inj (scalarRead.symm.trans operandRead)).2
    subst actualCoordinates
    subst stored
    exact ⟨reads.2.2, rfl, rfl⟩

end Merlin.GraphLoopsMemorySCF.After
