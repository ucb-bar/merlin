/- UNCOMPILED. Existing candidate epoch accesses use logical SSA coordinates
but physical slot addresses. Actual target accesses use mapped descriptors.
The same partial heap reads/writes and full caller world are derived below.
-/
import Merlin.GraphLoopsMemorySCFAfterScope
import Merlin.GraphLoopsShapeEpochMemorySteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem candidate_load_iff {Value User : Type} (bound : Nat) (specifications : Epoch.Specifications)
    (scalar : Epoch.BinaryMeaning Value) (mapping : Aliases) (key : String) (access : Epoch.Access) (result : Binding)
    (selected : Epoch.specAt specifications key = some (.load access result))
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation bound specifications mapping scalar) true true
      (.load access.object key) before after ↔
      ∃ coordinates value, Epoch.ReadCoordinates access before.observed.registers coordinates ∧
        before.heap (GraphLoopsReuse.slotOf objects access.object)
          (GraphLoopsReuse.Geometry.flatten access.shape coordinates) = some value ∧
        result.Holds (.scalar access.element value) ∧
        after = ⟨Epoch.writeRegister before.observed result.name (.scalar access.element value), before.heap⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary _ run =>
      cases run with
      | load object index key initial value observed address inside read action =>
        obtain ⟨a, r, coordinates, found, coordinatesRead, indexEqual⟩ := address
        rw [selected] at found
        cases Option.some.inj found
        obtain ⟨a, r, found, typed, final⟩ := action
        rw [selected] at found
        cases Option.some.inj found
        subst index
        subst observed
        exact ⟨coordinates, value, coordinatesRead, read, typed, rfl⟩
  · rintro ⟨coordinates, value, coordinatesRead, read, typed, rfl⟩
    apply GraphLoopsMemorySCF.Step.ordinary (by trivial)
    apply GraphLoopsReuse.Step.load access.object (GraphLoopsReuse.Geometry.flatten access.shape coordinates) key before value
    · exact ⟨access, result, coordinates, selected, coordinatesRead, rfl⟩
    · have inside := GraphLoopsReuse.Geometry.coordinate_inside coordinatesRead.2.2
      simpa only [GraphLoopsReuse.extentOf, GraphLoopsReuse.Object.extent_geometry, ← shape] using inside
    · exact read
    · exact ⟨access, result, selected, typed, rfl⟩

theorem candidate_store_iff {Value User : Type} (bound : Nat) (specifications : Epoch.Specifications)
    (scalar : Epoch.BinaryMeaning Value) (mapping : Aliases) (key : String) (access : Epoch.Access) (operand : Binding)
    (selected : Epoch.specAt specifications key = some (.store access operand))
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation bound specifications mapping scalar) true true
      (.store access.object key) before after ↔
      ∃ coordinates value, Epoch.ReadCoordinates access before.observed.registers coordinates ∧
        before.observed.registers operand.name = .scalar access.element value ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap
          (GraphLoopsReuse.slotOf objects access.object) (GraphLoopsReuse.Geometry.flatten access.shape coordinates) value⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary _ run =>
      cases run with
      | store object index key initial value computed inside =>
        obtain ⟨a, r, coordinates, found, coordinatesRead, scalarRead, indexEqual⟩ := computed
        rw [selected] at found
        cases Option.some.inj found
        subst index
        exact ⟨coordinates, value, coordinatesRead, scalarRead, rfl⟩
  · rintro ⟨coordinates, value, coordinatesRead, scalarRead, rfl⟩
    apply GraphLoopsMemorySCF.Step.ordinary (by trivial)
    apply GraphLoopsReuse.Step.store access.object (GraphLoopsReuse.Geometry.flatten access.shape coordinates) key before value
    · exact ⟨access, operand, coordinates, selected, coordinatesRead, scalarRead, rfl⟩
    · have inside := GraphLoopsReuse.Geometry.coordinate_inside coordinatesRead.2.2
      simpa only [GraphLoopsReuse.extentOf, GraphLoopsReuse.Object.extent_geometry, ← shape] using inside

/-- Exact ordinary-load result, with arbitrary extra scratch registers. The
private read remains partial; this theorem never manufactures initialized data.
-/
theorem actual_load_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (bound clock object : Nat) (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (operation : String) (buffer result : Binding) (indices : List Binding)
    (shape coordinates : List Nat) (element : String)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (load : meaning.action operation = Load)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value)
    (bufferValue : actual buffer.name = Epoch.view (GraphLoopsReuse.slotOf objects object) shape element)
    (indexValues : readArguments false [] actual indices = coordinates.map Runtime.index)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation (buffer :: indices) [result]).step meaning actual
      (Epoch.clockWorld bound clock before) next world ↔
      ∃ value, GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
        before.heap (GraphLoopsReuse.slotOf objects object) (GraphLoopsReuse.Geometry.flatten shape coordinates) = some value ∧
        result.Holds (.scalar element value) ∧
        next = GraphLoopsShape.put actual result.name (.scalar element value) ∧
        world = Epoch.clockWorld bound clock before := by
  have arguments : readArguments false [] actual (buffer :: indices) =
      Epoch.view (GraphLoopsReuse.slotOf objects object) shape element :: coordinates.map Runtime.index := by
    change actual buffer.name :: readArguments false [] actual indices = _
    rw [bufferValue, indexValues]
  simp only [Epoch.Atom.step, load, arguments, Epoch.load_iff]
  constructor
  · rintro ⟨values, ⟨value, inside, read, rfl, rfl⟩, typed, assigned⟩
    have privateRead := (Epoch.read_private bound (GraphLoopsReuse.slotOf objects object) _
      before.heap before.observed.callerHeap slotInside).symm.trans read
    have holds : result.Holds (.scalar element value) := by cases typed with | cons holds tail => exact holds
    exact ⟨value, inside, privateRead, holds, assigned, rfl⟩
  · rintro ⟨value, inside, read, holds, assigned, rfl⟩
    refine ⟨[.scalar element value], ⟨value, inside, ?_, rfl, rfl⟩, .cons holds .nil, assigned⟩
    exact (Epoch.read_private bound (GraphLoopsReuse.slotOf objects object) _
      before.heap before.observed.callerHeap slotInside).trans read

theorem actual_store_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (bound clock object : Nat) (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (operation : String) (operand buffer : Binding) (indices : List Binding)
    (shape coordinates : List Nat) (element : String) (value : Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (store : meaning.action operation = Store)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value)
    (scalarValue : actual operand.name = .scalar element value)
    (bufferValue : actual buffer.name = Epoch.view (GraphLoopsReuse.slotOf objects object) shape element)
    (indexValues : readArguments false [] actual indices = coordinates.map Runtime.index)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation (operand :: buffer :: indices) []).step meaning actual
      (Epoch.clockWorld bound clock before) next world ↔
      GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧ next = actual ∧
        world = Epoch.clockWorld bound clock
          ⟨before.observed, GraphLoopsReuse.replaceCell before.heap (GraphLoopsReuse.slotOf objects object)
            (GraphLoopsReuse.Geometry.flatten shape coordinates) value⟩ := by
  have arguments : readArguments false [] actual (operand :: buffer :: indices) =
      .scalar element value :: Epoch.view (GraphLoopsReuse.slotOf objects object) shape element ::
        coordinates.map Runtime.index := by
    change actual operand.name :: actual buffer.name :: readArguments false [] actual indices = _
    rw [scalarValue, bufferValue, indexValues]
  simp only [Epoch.Atom.step, store, arguments, Epoch.store_iff]
  have frame := Epoch.clock_private_store bound clock (GraphLoopsReuse.slotOf objects object)
    (GraphLoopsReuse.Geometry.flatten shape coordinates) value before slotInside
  constructor
  · rintro ⟨values, ⟨inside, rfl, world⟩, _, assigned⟩
    exact ⟨inside, assigned, world.trans frame.symm⟩
  · rintro ⟨inside, assigned, world⟩
    exact ⟨[], ⟨inside, rfl, world.trans frame⟩, .nil, assigned⟩

end Merlin.GraphLoopsMemorySCF.After
