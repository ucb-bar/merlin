/- Constructive nonzero initializer witnesses. These instantiate the common
abstract operation semantics, not native scalar/allocator conformance. Bounds
remain parameters; no shape-sized proof expansion is used by the validator. -/
import Merlin.GraphLoopsMemorySCFModel

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.Witness

def uniform (rows columns outerIV innerIV seed : Nat) : Initializer :=
  ⟨[rows, columns], [outerIV, innerIV], [], ⟨[], seed, [outerIV, innerIV]⟩⟩

/-- Totality is DERIVED here for explicit constant-store witness bodies only;
it is never inferred for the arbitrary partial meanings in the main theorem. -/
theorem scf_exists {State : Type} {body : Nat → State → State → Prop} (upper : Nat)
    (total : ∀ index, index < upper → ∀ before, ∃ after, body index before after) :
    ∀ count lower before, upper ≤ lower + count →
      ∃ trace after, GraphLoopsSegment.Execution body upper 1 lower before trace after := by
  intro count
  induction count with
  | zero =>
    intro lower before stopped
    exact ⟨[], before, .done before (by omega)⟩
  | succ count induction =>
    intro lower before bound
    by_cases stopped : upper ≤ lower
    · exact ⟨[], before, .done before stopped⟩
    · have inside : lower < upper := by omega
      obtain ⟨middle, first⟩ := total lower inside before
      obtain ⟨trace, after, rest⟩ := induction (lower + 1) middle (by omega)
      exact ⟨lower :: trace, after, .more inside first rest⟩

theorem scf_invariant {State : Type} {body : Nat → State → State → Prop} (invariant : State → Prop)
    (preserve : ∀ index before after, body index before after → invariant before → invariant after)
    {upper lower stride : Nat} {before after : State} {trace : List Nat}
    (run : GraphLoopsSegment.Execution body upper stride lower before trace after) (initial : invariant before) : invariant after := by
  induction run with
  | done => exact initial
  | more _ first _ induction => exact induction (preserve _ _ _ first initial)

theorem leaf_result {Value : Type} (meaning : PureMeaning Value) (outerIV innerIV seed row column : Nat)
    (different : outerIV ≠ innerIV) (outerFresh : seed ≠ outerIV) (innerFresh : seed ≠ innerIV)
    (captures : Env Value) (value : Value) (captured : captures seed = some (.scalar value)) :
    LeafResult meaning ⟨[], seed, [outerIV, innerIV]⟩
      (put (put captures outerIV (.index row)) innerIV (.index column)) [row, column] value := by
  refine ⟨_, .done _, ?_, ?_⟩
  · simp only [put, ite_eq_right innerFresh, ite_eq_right outerFresh, captured]
  · simp only [read, put, ite_eq_right different, ↓reduceIte, List.map_cons, List.map_nil,
      Option.pure_def]
    rfl

def AllValues {Value : Type} (value : Value) (writes : Log Value) : Prop := ∀ event ∈ writes, event.2 = value

theorem uniform_leaf_values {Value : Type} {meaning : PureMeaning Value} {rows columns iv seed index : Nat}
    (fresh : seed ≠ iv) {captures : Env Value} {value : Value} (captured : captures seed = some (.scalar value))
    {before after : Log Value} {indices : List Nat}
    (run : LeafBody meaning ⟨[], seed, indices⟩ [rows, columns] iv captures index before after)
    (initial : AllValues value before) : AllValues value after := by
  obtain ⟨emitted, stored, rfl⟩ := run
  obtain ⟨coordinates, storedValue, ⟨env, evaluated, scalar, _indices⟩, _inside, rfl⟩ := stored
  cases evaluated
  have same : storedValue = value := by
    simp only [put, ite_eq_right fresh, captured] at scalar
    exact (Datum.scalar.inj (Option.some.inj scalar)).symm
  subst storedValue
  intro event member
  rcases List.mem_append.mp member with previous | last
  · exact initial event previous
  · have equal : event = (GraphLoopsReuse.Geometry.flatten [rows, columns] coordinates, value) := List.mem_singleton.mp last
    exact congrArg Prod.snd equal

theorem uniform_run_exists {Value : Type} (meaning : PureMeaning Value) (rows columns outerIV innerIV seed tile : Nat)
    (different : outerIV ≠ innerIV) (outerFresh : seed ≠ outerIV) (innerFresh : seed ≠ innerIV)
    (captures : Env Value) (value : Value) (captured : captures seed = some (.scalar value)) :
    ∃ writes, Run meaning (uniform rows columns outerIV innerIV seed) tile false captures writes := by
  have total : ∀ row, row < rows → ∀ before,
      ∃ after, NestedBody meaning [] ⟨[], seed, [outerIV, innerIV]⟩ rows columns outerIV innerIV captures row before after := by
    intro row rowInside before
    have leaves : ∀ column, column < columns → ∀ writes,
        ∃ after, LeafBody meaning ⟨[], seed, [outerIV, innerIV]⟩ [rows, columns] innerIV
          (put captures outerIV (.index row)) column writes after := by
      intro column columnInside writes
      refine ⟨writes ++ [(row * columns + column, value)], ?_⟩
      refine ⟨⟨_, _⟩, ⟨[row, column], value, leaf_result meaning outerIV innerIV seed row column different outerFresh innerFresh captures value captured,
        ⟨rowInside, columnInside, True.intro⟩, rfl⟩, ?_⟩
      simp only [GraphLoopsReuse.Geometry.flatten, GraphLoopsReuse.Geometry.volume, Nat.mul_one, Nat.add_zero]
    obtain ⟨trace, after, inner⟩ := scf_exists columns leaves columns 0 before (by omega)
    exact ⟨after, _, trace, .done _, inner⟩
  obtain ⟨trace, writes, run⟩ := scf_exists rows total rows 0 [] (by omega)
  exact ⟨writes, trace, run⟩

theorem uniform_run_values {Value : Type} {meaning : PureMeaning Value} {rows columns outerIV innerIV seed tile : Nat}
    (outerFresh : seed ≠ outerIV) (innerFresh : seed ≠ innerIV)
    {captures : Env Value} {value : Value} (captured : captures seed = some (.scalar value)) {writes : Log Value}
    (run : Run meaning (uniform rows columns outerIV innerIV seed) tile false captures writes) : AllValues value writes := by
  obtain ⟨_trace, run⟩ := run
  have source : GraphLoopsSegment.Execution
      (NestedBody meaning [] ⟨[], seed, [outerIV, innerIV]⟩ rows columns outerIV innerIV captures) rows 1 0 [] _ writes := run
  apply scf_invariant (AllValues value) (fun row before after body initial => ?_) source (by simp [AllValues])
  obtain ⟨rowEnv, _trace, computed, inner⟩ := body
  cases computed
  have seedValue : put captures outerIV (.index row) seed = some (.scalar value) := (ite_eq_right outerFresh).trans captured
  exact scf_invariant (AllValues value) (fun _ _ _ leaf initial => uniform_leaf_values innerFresh seedValue leaf initial) inner initial

theorem writes_preserve_uniform {Value : Type} {value : Value} {writes : Log Value}
    (uniformValues : AllValues value writes) (cells : Cells Value) {index : Nat}
    (initialized : cells index = some value) : applyWrites writes cells index = some value := by
  induction writes generalizing cells with
  | nil => exact initialized
  | cons event rest induction =>
    have eventValue := uniformValues event (List.mem_cons_self)
    have restValues : AllValues value rest := fun write member => uniformValues write (List.mem_cons_of_mem event member)
    apply induction restValues
    change (if index = event.1 then some event.2 else cells index) = some value
    split
    · exact congrArg some eventValue
    · exact initialized

theorem writes_uniform {Value : Type} {value : Value} {writes : Log Value} (uniformValues : AllValues value writes)
    (cells : Cells Value) {index : Nat} (written : ∃ stored, (index, stored) ∈ writes) :
    applyWrites writes cells index = some value := by
  induction writes generalizing cells with
  | nil => obtain ⟨_, member⟩ := written; cases member
  | cons event rest induction =>
    have eventValue := uniformValues event (List.mem_cons_self)
    have restValues : AllValues value rest := fun write member => uniformValues write (List.mem_cons_of_mem event member)
    obtain ⟨stored, member⟩ := written
    rcases List.mem_cons.mp member with equal | later
    · subst event
      have same : stored = value := eventValue
      subst stored
      exact writes_preserve_uniform restValues _ (by simp only [writeCell, ↓reduceIte])
    · exact induction restValues _ ⟨stored, later⟩

/-- Explicit successful operational initializer with EVERY cell equal to the
nonzero captured seed, starting from arbitrary prior private contents. -/
theorem uniform_heap {Value : Type} (meaning : PureMeaning Value) (rows columns outerIV innerIV seed tile object : Nat)
    (positiveRows : 0 < rows) (positiveColumns : 0 < columns)
    (different : outerIV ≠ innerIV) (outerFresh : seed ≠ outerIV) (innerFresh : seed ≠ innerIV)
    (captures : Env Value) (value : Value) (captured : captures seed = some (.scalar value)) (before : Heap Value) :
    ∃ after, HeapRun meaning (uniform rows columns outerIV innerIV seed) tile object false captures before after ∧
      ∀ index, index < rows * columns → after object index = some value := by
  have valid : (uniform rows columns outerIV innerIV seed).WellFormed := by
    refine ⟨Or.inr ?_, rfl, ?_, ?_⟩
    · exact ⟨rfl, rfl, positiveRows, positiveColumns, different⟩
    · simp [uniform, definitions]
    · simp [uniform, definitions]
  obtain ⟨writes, run⟩ := uniform_run_exists meaning rows columns outerIV innerIV seed tile different outerFresh innerFresh captures value captured
  have allValues := uniform_run_values outerFresh innerFresh captured run
  refine ⟨applyHeapWrites object writes before, (heap_run_iff_log meaning _ tile object captures before _ valid).mpr ⟨writes, run, rfl⟩, ?_⟩
  intro index inside
  rw [applyHeapWrites_at]
  apply writes_uniform allValues
  apply run.covers valid index
  simpa only [uniform, GraphLoopsReuse.Geometry.volume, Nat.mul_one] using inside

end Merlin.GraphLoopsMemorySCF.Witness
