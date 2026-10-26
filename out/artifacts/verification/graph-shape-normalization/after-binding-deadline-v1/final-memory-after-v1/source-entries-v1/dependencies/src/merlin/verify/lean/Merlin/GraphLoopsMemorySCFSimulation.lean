/- Complete initializer stores related across allocation-changing worlds.
The exact stored values follow from coverage and the operational/log theorem.
Only initialized live cells are compared; stale and out-of-domain cells are not. -/
import Merlin.GraphLoopsMemorySCFOperational

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

theorem heap_run_iff_log_mode {Value : Type} (meaning : PureMeaning Value) (program : Initializer)
    (tile object : Nat) (segmented : Bool) (captures : Env Value) (before after : Heap Value)
    (valid : program.WellFormed) (positive : 0 < tile) :
    HeapRun meaning program tile object segmented captures before after ↔
      ∃ writes, Run meaning program tile false captures writes ∧ after = applyHeapWrites object writes before := by
  cases segmented with
  | false => exact heap_run_iff_log meaning program tile object captures before after valid
  | true =>
    exact (segmentation_exact_heaps meaning program tile object positive captures before after).symm.trans
      (heap_run_iff_log meaning program tile object captures before after valid)

/-- Replace the SAME-WORLD atomic fill by explicit ordered stores. Only
in-bounds equality is asserted; atomic fill clears out-of-range cells whereas
actual stores preserve them. Other private objects and public state are framed. -/
theorem materializes_in_bounds {Value : Type} {meaning : PureMeaning Value} {program : Initializer}
    {tile : Nat} (valid : program.WellFormed) {captures : Env Value} {writes : Log Value}
    (run : Run meaning program tile false captures writes) (object : Nat) (heap : Heap Value) :
    ∃ values : Nat → Value, (∀ index, index < GraphLoopsReuse.Geometry.volume program.shape →
      applyHeapWrites object writes heap object index =
        GraphLoopsReuse.initializedCells (GraphLoopsReuse.Geometry.volume program.shape) values index) ∧
      ∀ other, other ≠ object → applyHeapWrites object writes heap other = heap other := by
  obtain ⟨values, initialized⟩ := run.materializes valid
  refine ⟨values, ?_, fun _ different => applyHeapWrites_other object _ writes heap different⟩
  intro index inside
  rw [applyHeapWrites_at, initialized _ _ inside]
  exact (ite_eq_left inside).symm

/-- Full live-epoch relation after complete initialization, for arbitrary stale
source and target heaps and arbitrary partial scalar operation meanings. -/
theorem related_initialize_logs {Value Public : Type} {objects : GraphLoopsReuse.Objects} {ready : List Nat}
    {source target : GraphLoopsReuse.Memory Value Public}
    (related : GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) ready source target)
    {object : Nat} (separate : ∀ other ∈ ready, other ≠ object → GraphLoopsReuse.slotOf objects other ≠ GraphLoopsReuse.slotOf objects object)
    {meaning : PureMeaning Value} {program : Initializer} {tile : Nat} (valid : program.WellFormed)
    (shape : program.shape = (GraphLoopsReuse.objectAt objects object).shape) {captures : Env Value} {writes : Log Value}
    (run : Run meaning program tile false captures writes) :
    GraphLoopsReuse.Related (GraphLoopsReuse.slotOf objects) (GraphLoopsReuse.extentOf objects) (object :: ready)
      ⟨source.observed, applyHeapWrites object writes source.heap⟩
      ⟨target.observed, applyHeapWrites (GraphLoopsReuse.slotOf objects object) writes target.heap⟩ := by
  obtain ⟨values, materialized⟩ := run.materializes valid
  have volume : GraphLoopsReuse.Geometry.volume program.shape = GraphLoopsReuse.extentOf objects object := by
    rw [shape]
    exact (GraphLoopsReuse.Object.extent_geometry _).symm
  refine ⟨related.1, ?_⟩
  intro other initialized index inside
  change applyHeapWrites object writes source.heap other index =
    applyHeapWrites (GraphLoopsReuse.slotOf objects object) writes target.heap (GraphLoopsReuse.slotOf objects other) index
  rcases List.mem_cons.mp initialized with rfl | previous
  · rw [applyHeapWrites_at, applyHeapWrites_at]
    exact (materialized _ _ (volume ▸ inside)).trans (materialized _ _ (volume ▸ inside)).symm
  · by_cases equal : other = object
    · subst other
      rw [applyHeapWrites_at, applyHeapWrites_at]
      exact (materialized _ _ (volume ▸ inside)).trans (materialized _ _ (volume ▸ inside)).symm
    · rw [applyHeapWrites_other object other writes source.heap equal,
        applyHeapWrites_other _ _ writes target.heap (separate other previous equal)]
      exact related.read previous inside

end Merlin.GraphLoopsMemorySCF
