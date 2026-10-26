/- Operational initializer framing over the entire heap. The old private epoch
model and the typed descriptor model agree on each ordered store, including all
caller objects and all untouched private cells. No result-level assumption. -/
import Merlin.GraphLoopsShapeEpochPublic

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem merge_heap_writes {Value : Type} (bound object : Nat) (inside : object < bound)
    (writes : GraphLoopsMemorySCF.Log Value) (privateHeap caller : Heap Value) :
    mergeHeap bound (GraphLoopsMemorySCF.applyHeapWrites object writes privateHeap) caller =
      GraphLoopsMemorySCF.applyHeapWrites object writes (mergeHeap bound privateHeap caller) := by
  induction writes generalizing privateHeap with
  | nil => rfl
  | cons event rest induction =>
    change mergeHeap bound (GraphLoopsMemorySCF.applyHeapWrites object rest
      (GraphLoopsReuse.replaceCell privateHeap object event.1 event.2)) caller = _
    rw [induction, merge_private_cell bound object event.1 event.2 privateHeap caller inside]
    rfl

/-- A full-heap initializer run has a corresponding private-heap run, and vice
versa, with the SAME concrete final full heap. Caller writes cannot disappear. -/
theorem heap_run_partition {Value : Type} (bound object tile : Nat) (inside : object < bound)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (program : GraphLoopsMemorySCF.Initializer)
    (valid : program.WellFormed) (captures : GraphLoopsMemorySCF.Env Value)
    (privateHeap caller after : Heap Value) :
    GraphLoopsMemorySCF.HeapRun meaning program tile object false captures
      (mergeHeap bound privateHeap caller) after ↔
      ∃ next, GraphLoopsMemorySCF.HeapRun meaning program tile object false captures privateHeap next ∧
        after = mergeHeap bound next caller := by
  rw [GraphLoopsMemorySCF.heap_run_iff_log meaning program tile object captures _ _ valid]
  constructor
  · rintro ⟨writes, run, rfl⟩
    refine ⟨GraphLoopsMemorySCF.applyHeapWrites object writes privateHeap, ?_, ?_⟩
    · exact (GraphLoopsMemorySCF.heap_run_iff_log meaning program tile object captures _ _ valid).mpr ⟨writes, run, rfl⟩
    · exact (merge_heap_writes bound object inside writes privateHeap caller).symm
  · rintro ⟨next, run, rfl⟩
    obtain ⟨writes, logRun, rfl⟩ :=
      (GraphLoopsMemorySCF.heap_run_iff_log meaning program tile object captures _ _ valid).mp run
    exact ⟨writes, logRun, merge_heap_writes bound object inside writes privateHeap caller⟩

/-- This final-world form preserves public observations as well as caller cells.
The register ghost state is framed because these initializer bodies only store. -/
theorem initializer_world_partition {Value User : Type} (bound object tile : Nat) (inside : object < bound)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (program : GraphLoopsMemorySCF.Initializer)
    (valid : program.WellFormed) (captures : GraphLoopsMemorySCF.Env Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) (after : MemoryWorld Value User) :
    (after.observed = before.observed.observed ∧
      GraphLoopsMemorySCF.HeapRun meaning program tile object false captures (fullWorld bound before).heap after.heap) ↔
      ∃ next, GraphLoopsMemorySCF.HeapRun meaning program tile object false captures before.heap next ∧
        after = fullWorld bound ⟨before.observed, next⟩ := by
  change (after.observed = before.observed.observed ∧
    GraphLoopsMemorySCF.HeapRun meaning program tile object false captures
      (mergeHeap bound before.heap before.observed.callerHeap) after.heap) ↔ _
  rw [heap_run_partition bound object tile inside meaning program valid captures before.heap before.observed.callerHeap]
  constructor
  · rintro ⟨observed, next, run, heap⟩
    refine ⟨next, run, ?_⟩
    cases after
    cases observed
    cases heap
    rfl
  · rintro ⟨next, run, rfl⟩
    exact ⟨rfl, next, run, rfl⟩

end Merlin.GraphLoopsShape.Epoch
