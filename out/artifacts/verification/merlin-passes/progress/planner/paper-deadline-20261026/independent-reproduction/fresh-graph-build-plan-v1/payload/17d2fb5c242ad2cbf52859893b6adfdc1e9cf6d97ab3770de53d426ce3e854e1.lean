/- Compositional meaning of private store logs. Ordered writes are applied one
at a time; public state and other objects are framed. Values are derived from
successful complete loop executions, not an admitted initializer summary. -/
import Merlin.GraphLoopsMemorySCFRect

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF

abbrev Cells (Value : Type) := GraphLoopsReuse.Cells Value
abbrev Heap (Value : Type) := GraphLoopsReuse.Heap Value

def writeCell {Value : Type} (cells : Cells Value) (event : Write Value) : Cells Value :=
  fun index => if index = event.1 then some event.2 else cells index

def applyWrites {Value : Type} (writes : Log Value) (cells : Cells Value) : Cells Value :=
  writes.foldl writeCell cells

def applyHeapWrites {Value : Type} (object : Nat) (writes : Log Value) (heap : Heap Value) : Heap Value :=
  writes.foldl (fun before event => GraphLoopsReuse.replaceCell before object event.1 event.2) heap

theorem applyWrites_append {Value : Type} (first second : Log Value) (cells : Cells Value) :
    applyWrites (first ++ second) cells = applyWrites second (applyWrites first cells) := by
  exact List.foldl_append

theorem applyHeapWrites_append {Value : Type} (object : Nat) (first second : Log Value) (heap : Heap Value) :
    applyHeapWrites object (first ++ second) heap = applyHeapWrites object second (applyHeapWrites object first heap) := by
  exact List.foldl_append

theorem applyWrites_agree {Value : Type} (writes : Log Value) (first second : Cells Value) (index : Nat)
    (agree : first index = second index) : applyWrites writes first index = applyWrites writes second index := by
  induction writes generalizing first second with
  | nil => exact agree
  | cons event rest induction =>
    apply induction
    unfold writeCell
    split
    · rfl
    · exact agree

theorem applyWrites_some {Value : Type} (writes : Log Value) (cells : Cells Value) (index : Nat)
    (initialized : ∃ value, cells index = some value) : ∃ value, applyWrites writes cells index = some value := by
  induction writes generalizing cells with
  | nil => exact initialized
  | cons event rest induction =>
    apply induction
    by_cases equal : index = event.1
    · exact ⟨event.2, ite_eq_left equal⟩
    · obtain ⟨value, cell⟩ := initialized
      exact ⟨value, (ite_eq_right equal).trans cell⟩

theorem applyWrites_written {Value : Type} (writes : Log Value) (cells : Cells Value) {index : Nat}
    (written : ∃ value, (index, value) ∈ writes) : ∃ value, applyWrites writes cells index = some value := by
  induction writes generalizing cells with
  | nil => obtain ⟨_value, member⟩ := written; cases member
  | cons event rest induction =>
    obtain ⟨value, member⟩ := written
    rcases List.mem_cons.mp member with equal | later
    · subst event
      exact applyWrites_some rest (writeCell cells (index, value)) index ⟨value, by simp only [writeCell, ↓reduceIte]⟩
    · exact induction _ ⟨value, later⟩

theorem applyWrites_independent {Value : Type} (writes : Log Value) (first second : Cells Value) {index : Nat}
    (written : ∃ value, (index, value) ∈ writes) : applyWrites writes first index = applyWrites writes second index := by
  induction writes generalizing first second with
  | nil => obtain ⟨_value, member⟩ := written; cases member
  | cons event rest induction =>
    obtain ⟨value, member⟩ := written
    rcases List.mem_cons.mp member with equal | later
    · subst event
      exact applyWrites_agree rest _ _ index (by simp only [writeCell, ↓reduceIte])
    · exact induction _ _ ⟨value, later⟩

theorem applyHeapWrites_at {Value : Type} (object : Nat) (writes : Log Value) (heap : Heap Value) (index : Nat) :
    applyHeapWrites object writes heap object index = applyWrites writes (heap object) index := by
  induction writes generalizing heap with
  | nil => rfl
  | cons event rest induction =>
    change applyHeapWrites object rest (GraphLoopsReuse.replaceCell heap object event.1 event.2) object index =
      applyWrites rest (writeCell (heap object) event) index
    rw [induction]
    have objectCell : GraphLoopsReuse.replaceCell heap object event.1 event.2 object = writeCell (heap object) event := by
      funext other
      simp only [GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, writeCell, ↓reduceIte]
    rw [objectCell]

theorem applyHeapWrites_other {Value : Type} (object other : Nat) (writes : Log Value) (heap : Heap Value)
    (different : other ≠ object) : applyHeapWrites object writes heap other = heap other := by
  induction writes generalizing heap with
  | nil => rfl
  | cons event rest induction =>
    exact (induction _).trans (ite_eq_right different)

/-- Every successful complete run induces scalar values for ALL valid cells,
independently of arbitrary prior/stale contents. Choice is derived from writes. -/
theorem Run.materializes {Value : Type} {meaning : PureMeaning Value} {program : Initializer} {tile : Nat}
    (valid : program.WellFormed) {captures : Env Value} {writes : Log Value}
    (run : Run meaning program tile false captures writes) :
    ∃ values : Nat → Value, ∀ cells index, index < GraphLoopsReuse.Geometry.volume program.shape →
      applyWrites writes cells index = some (values index) := by
  have coverage := run.covers valid
  have initialized : ∀ index, index < GraphLoopsReuse.Geometry.volume program.shape →
      ∃ value, applyWrites writes (fun _ => none) index = some value :=
    fun index inside => applyWrites_written writes (fun _ => none) (coverage index inside)
  obtain ⟨first, _firstInitialized⟩ := initialized 0 (Initializer.positive_volume valid)
  let values : Nat → Value := fun index => if inside : index < GraphLoopsReuse.Geometry.volume program.shape then
    Classical.choose (initialized index inside) else first
  refine ⟨values, ?_⟩
  intro cells index inside
  rw [applyWrites_independent writes cells (fun _ => none) (coverage index inside)]
  exact (by simpa only [values, dite_eq_left inside] using Classical.choose_spec (initialized index inside))

end Merlin.GraphLoopsMemorySCF
