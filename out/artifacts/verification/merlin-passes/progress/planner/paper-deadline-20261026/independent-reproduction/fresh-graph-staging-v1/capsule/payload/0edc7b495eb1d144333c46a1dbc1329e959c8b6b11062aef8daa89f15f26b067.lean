/- Logical allocation naming for the typed-to-epoch adapter. Allocation order is
explicit private semantic state, not inferred from identical operation keys.
Native allocation failures, addresses and resource usage are not modeled here. -/
import Merlin.GraphLoopsShapeEpochFrame
import Merlin.GraphLoopsShapeEpochScope

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure AllocationPublic (User : Type) where
  next : Nat
  observed : User

/-- The program-point allocation count reconstructs private naming state. It is
not included in the caller observation or the old epoch PublicState equality. -/
def clockWorld {Value User : Type} (bound next : Nat)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) : MemoryWorld Value (AllocationPublic User) :=
  ⟨⟨next, memory.observed.observed⟩, mergeHeap bound memory.heap memory.observed.callerHeap⟩

def Allocate {Value User : Type} (shape : List Nat) (element : String)
    (inputs : List (Runtime Value)) (before : MemoryWorld Value (AllocationPublic User))
    (outputs : List (Runtime Value)) (after : MemoryWorld Value (AllocationPublic User)) : Prop :=
  inputs = [] ∧ outputs = [view before.observed.next shape element] ∧
    after = ⟨⟨before.observed.next + 1, before.observed.observed⟩,
      GraphLoopsReuse.replaceObject before.heap before.observed.next (fun _ => none)⟩

def Release {Value User : Type} (inputs : List (Runtime Value))
    (before : MemoryWorld Value (AllocationPublic User)) (outputs : List (Runtime Value))
    (after : MemoryWorld Value (AllocationPublic User)) : Prop :=
  ∃ type descriptor valid, inputs = [Runtime.view type descriptor valid] ∧ outputs = [] ∧
    after = ⟨before.observed, GraphLoopsReuse.replaceObject before.heap descriptor.allocation (fun _ => none)⟩

/-- The same allocation meaning serves identical operation keys at every site.
A checked source allocation ordinal equals the pre-step counter. -/
theorem allocation_epoch {Value User : Type} (bound object : Nat) (inside : object < bound)
    (shape : List Nat) (element : String)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) (objects : GraphLoopsReuse.Objects)
    (retained : Bool) (outputs : List (Runtime Value)) (after : MemoryWorld Value (AllocationPublic User)) :
    Allocate shape element [] (clockWorld bound object memory) outputs after ↔
      outputs = [view object shape element] ∧
        after = clockWorld bound (object + 1) (GraphLoopsReuse.clearMemory objects false object retained memory) := by
  simp only [Allocate, clockWorld, true_and, GraphLoopsReuse.clearMemory, Bool.false_and,
    Bool.false_eq_true, ↓reduceIte, GraphLoopsReuse.addressOf]
  rw [merge_private_clear bound object memory.heap memory.observed.callerHeap inside]

theorem release_epoch {Value User : Type} (bound next object : Nat) (inside : object < bound)
    (shape : List Nat) (element : String)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) (objects : GraphLoopsReuse.Objects)
    (retained : Bool) (outputs : List (Runtime Value)) (after : MemoryWorld Value (AllocationPublic User)) :
    Release [view object shape element] (clockWorld bound next memory) outputs after ↔
      outputs = [] ∧ after = clockWorld bound next (GraphLoopsReuse.clearMemory objects false object retained memory) := by
  constructor
  · rintro ⟨actual, descriptor, valid, inputs, outputs, after⟩
    have descriptors := Runtime.view.inj (List.cons.inj inputs).1
    have sameDescriptor : descriptor = Epoch.descriptor object shape element := descriptors.2.symm
    subst descriptor
    refine ⟨outputs, ?_⟩
    rw [after]
    simp only [clockWorld, Epoch.descriptor, GraphLoopsReuse.clearMemory, Bool.false_and,
      Bool.false_eq_true, ↓reduceIte, GraphLoopsReuse.addressOf]
    rw [merge_private_clear bound object memory.heap memory.observed.callerHeap inside]
  · rintro ⟨rfl, rfl⟩
    refine ⟨type shape element, descriptor object shape element, conforms object shape element, rfl, rfl, ?_⟩
    simp only [clockWorld, descriptor, GraphLoopsReuse.clearMemory, Bool.false_and,
      Bool.false_eq_true, ↓reduceIte, GraphLoopsReuse.addressOf]
    rw [merge_private_clear bound object memory.heap memory.observed.callerHeap inside]

/-- A future view ghost can be prebound without affecting any actual earlier
free read; the complete Code theorem covers nested captures and partial effects. -/
theorem preseed_fresh_view {Value World : Type} (meaning : Interpretation Value World) (code : Code)
    (aliases : Aliases) (env : Env Value) (name object : Nat) (shape : List Nat) (element : String)
    (unused : name ∉ freeReads code) (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases code env before outputs after ↔
      Eval meaning false aliases code (put env name (view object shape element)) before outputs after := by
  apply eval_free_reads
  intro other read
  have distinct : other ≠ name := fun equal => unused (equal ▸ read)
  exact (ite_eq_right distinct).symm

end Merlin.GraphLoopsShape.Epoch
