/- Derive private load/store instruction behavior from the single checked SSA
primitive interpretation. Bounds are derived from coordinates and actual object
shapes, rather than assumed as independent in-bounds oracle assertions. -/
import Merlin.GraphLoopsShapeEpochPrimitives

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem load_step_iff {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (access : Access) (result : Binding)
    (selected : specAt specifications key = some (.load access result))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
      (.load access.object key) before after ↔
      ∃ coordinates value, ReadCoordinates access before.observed.registers coordinates ∧
        before.heap access.object (GraphLoopsReuse.Geometry.flatten access.shape coordinates) = some value ∧
        result.Holds (.scalar access.element value) ∧
        after = ⟨writeRegister before.observed result.name (.scalar access.element value), before.heap⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run =>
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

theorem store_step_iff {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (access : Access) (operand : Binding)
    (selected : specAt specifications key = some (.store access operand))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (shape : access.shape = (GraphLoopsReuse.objectAt objects access.object).shape)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
      (.store access.object key) before after ↔
      ∃ coordinates value, ReadCoordinates access before.observed.registers coordinates ∧
        before.observed.registers operand.name = .scalar access.element value ∧
        after = ⟨before.observed, GraphLoopsReuse.replaceCell before.heap access.object
          (GraphLoopsReuse.Geometry.flatten access.shape coordinates) value⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run =>
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

end Merlin.GraphLoopsShape.Epoch
