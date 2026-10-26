/- Explicit reconstruction of typed full heaps from private epoch memory and
caller-visible memory. Registers are ghost SSA state, not pointer observations.
The caller/private allocation partition is an explicit local domain premise. -/
import Merlin.GraphLoopsShapeEpochAlpha

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure PublicState (Value User : Type) where
  registers : Env Value
  callerHeap : Heap Value
  observed : User
  returned : List (Runtime Value)

/-- Logical allocations below the private bound are private; caller descriptors
must name objects outside it. This is not a statement about native addresses. -/
def mergeHeap {Value : Type} (bound : Nat) (privateHeap caller : Heap Value) : Heap Value :=
  fun object index => if object < bound then privateHeap object index else caller object index

def fullWorld {Value User : Type} (bound : Nat) (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    MemoryWorld Value User :=
  ⟨memory.observed.observed, mergeHeap bound memory.heap memory.observed.callerHeap⟩

theorem read_private {Value : Type} (bound object index : Nat) (privateHeap caller : Heap Value)
    (inside : object < bound) : mergeHeap bound privateHeap caller object index = privateHeap object index := by
  simp only [mergeHeap, inside, ↓reduceIte]

theorem read_caller {Value : Type} (bound object index : Nat) (privateHeap caller : Heap Value)
    (outside : ¬ object < bound) : mergeHeap bound privateHeap caller object index = caller object index := by
  simp only [mergeHeap, outside, ↓reduceIte]

theorem merge_private_cell {Value : Type} (bound object index : Nat) (value : Value) (privateHeap caller : Heap Value)
    (inside : object < bound) :
    mergeHeap bound (GraphLoopsReuse.replaceCell privateHeap object index value) caller =
      GraphLoopsReuse.replaceCell (mergeHeap bound privateHeap caller) object index value := by
  funext other offset
  by_cases same : other = object
  · subst other
    simp only [mergeHeap, inside, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, ↓reduceIte]
  · simp only [mergeHeap, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, same, ↓reduceIte]

theorem merge_caller_cell {Value : Type} (bound object index : Nat) (value : Value) (privateHeap caller : Heap Value)
    (outside : ¬ object < bound) :
    mergeHeap bound privateHeap (GraphLoopsReuse.replaceCell caller object index value) =
      GraphLoopsReuse.replaceCell (mergeHeap bound privateHeap caller) object index value := by
  funext other offset
  by_cases same : other = object
  · subst other
    simp only [mergeHeap, outside, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, ↓reduceIte]
  · simp only [mergeHeap, GraphLoopsReuse.replaceCell, GraphLoopsReuse.replaceObject, same, ↓reduceIte]

theorem merge_private_clear {Value : Type} (bound object : Nat) (privateHeap caller : Heap Value)
    (inside : object < bound) :
    mergeHeap bound (GraphLoopsReuse.replaceObject privateHeap object (fun _ => none)) caller =
      GraphLoopsReuse.replaceObject (mergeHeap bound privateHeap caller) object (fun _ => none) := by
  funext other index
  by_cases same : other = object
  · subst other
    simp only [mergeHeap, inside, GraphLoopsReuse.replaceObject, ↓reduceIte]
  · simp only [mergeHeap, GraphLoopsReuse.replaceObject, same, ↓reduceIte]

/-- Updating private SSA ghost registers does not change the observable world. -/
theorem registers_frame {Value User : Type} (bound : Nat) (memory : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : Env Value) :
    fullWorld bound { memory with observed := { memory.observed with registers := registers } } = fullWorld bound memory := rfl

theorem returned_frame {Value User : Type} (bound : Nat) (memory : GraphLoopsReuse.Memory Value (PublicState Value User))
    (returned : List (Runtime Value)) :
    fullWorld bound { memory with observed := { memory.observed with returned := returned } } = fullWorld bound memory := rfl

theorem private_store_world {Value User : Type} (bound object index : Nat) (value : Value)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) (inside : object < bound) :
    fullWorld bound ⟨memory.observed, GraphLoopsReuse.replaceCell memory.heap object index value⟩ =
      ⟨memory.observed.observed, GraphLoopsReuse.replaceCell (fullWorld bound memory).heap object index value⟩ := by
  unfold fullWorld
  rw [← merge_private_cell bound object index value memory.heap memory.observed.callerHeap inside]

theorem caller_store_world {Value User : Type} (bound object index : Nat) (value : Value)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) (outside : ¬ object < bound) :
    fullWorld bound { memory with observed := { memory.observed with
      callerHeap := GraphLoopsReuse.replaceCell memory.observed.callerHeap object index value } } =
      ⟨memory.observed.observed, GraphLoopsReuse.replaceCell (fullWorld bound memory).heap object index value⟩ := by
  unfold fullWorld
  rw [← merge_caller_cell bound object index value memory.heap memory.observed.callerHeap outside]

end Merlin.GraphLoopsShape.Epoch
