/- Scoped SSA, explicit captures and private-store address binding.
No heap/public action is hidden inside a pure operator interpretation. -/
import Merlin.GraphLoopsMemorySCF

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

theorem read_agree {Value : Type} (first second : Env Value) (names : List Nat)
    (agree : ∀ name ∈ names, first name = second name) : read first names = read second names := by
  induction names with
  | nil => rfl
  | cons name rest induction =>
    have atName := agree name (List.mem_cons_self)
    have remaining := induction (fun other member => agree other (List.mem_cons_of_mem name member))
    simp only [read, atName, remaining]

theorem put_at {Value : Type} (env : Env Value) (name : Nat) (value : Datum Value) :
    put env name value name = some value := by simp only [put, ↓reduceIte]

theorem put_other {Value : Type} (env : Env Value) (name other : Nat) (value : Datum Value) (different : other ≠ name) :
    put env name value other = env other := ite_eq_right different

theorem LeafResult.coordinates {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {env : Env Value}
    {actual expected : List Nat} {value : Value} (run : LeafResult meaning leaf env actual value)
    (noShadow : ∀ name ∈ leaf.indices, name ∉ definitions leaf.ops)
    (binding : read env leaf.indices = some (expected.map Datum.index)) : actual = expected := by
  obtain ⟨after, evaluated, _storedValue, indices⟩ := run
  have frame := read_agree after env leaf.indices (fun name member => evaluated.frame (noShadow name member))
  rw [frame, binding] at indices
  have same : expected.map (Datum.index (Value := Value)) = actual.map Datum.index := Option.some.inj indices
  have injective : ∀ first second : Nat, Datum.index (Value := Value) first = Datum.index second → first = second := by
    intro first second equal
    cases equal
    rfl
  exact ((List.map_inj_right injective).mp same).symm

/-- A leaf private store appends exactly its SSA-selected coordinate/value.
The actual bound IVs, not a postulated whole-initializer result, fix the address. -/
theorem Emit.address {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {shape expected : List Nat}
    {before after : ScopedLog Value} (run : Emit meaning leaf shape before after)
    (noShadow : ∀ name ∈ leaf.indices, name ∉ definitions leaf.ops)
    (binding : read before.env leaf.indices = some (expected.map Datum.index)) :
    ∃ value, after.env = before.env ∧
      after.writes = before.writes ++ [(GraphLoopsReuse.Geometry.flatten shape expected, value)] := by
  obtain ⟨actual, value, computed, _inside, equal⟩ := run
  have coordinates := computed.coordinates noShadow binding
  subst actual
  subst after
  exact ⟨value, rfl, rfl⟩

theorem Emit.monotone {Value : Type} {meaning : PureMeaning Value} {leaf : Leaf} {shape : List Nat}
    {before after : ScopedLog Value} (run : Emit meaning leaf shape before after) :
    ∀ event ∈ before.writes, event ∈ after.writes := by
  obtain ⟨_coordinates, _value, _computed, _inside, rfl⟩ := run
  intro event member
  exact List.mem_append_left _ member

end Merlin.GraphLoopsMemorySCF
