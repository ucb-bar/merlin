/- UNCOMPILED source candidate, outside the authorized shape-after v2 cut.
Local allocation/release laws for the independently interpreted final program.
The physical allocator ordinal is an explicit obligation: logical object IDs
need not equal retained allocation ordinals. No native allocator claim follows.
-/
import Merlin.GraphLoopsMemorySCFAfterSlots
import Merlin.GraphLoopsShapeEpochAllocation

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem retained_clear {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (object : Nat) (before : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsReuse.clearMemory objects true object true before =
      GraphLoopsReuse.clearMemory objects false (GraphLoopsReuse.slotOf objects object) true before := by
  rfl

/-- A removed allocation/free performs no heap clear and no caller change.
The expanded typed target must separately bind its surviving alias descriptor.
-/
theorem skipped_clear {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (object : Nat) (before : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsReuse.clearMemory objects true object false before = before := by
  rfl

/-- Actual allocation returns the mapped descriptor and clears exactly its
physical slot in the full heap. `ordinal` must be checked at every retained site;
it is not an assertion that all slot labels form an allocation counter.
-/
theorem allocate_retained {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (bound object next : Nat) (logicalInside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (ordinal : next = GraphLoopsReuse.slotOf objects object)
    (shape : List Nat) (element : String)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (outputs : List (Runtime Value)) (after : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Epoch.Allocate shape element [] (Epoch.clockWorld bound next before) outputs after ↔
      outputs = [slotRuntime objects (Epoch.view object shape element)] ∧
      after = Epoch.clockWorld bound (next + 1)
        (GraphLoopsReuse.clearMemory objects true object true before) := by
  subst next
  rw [slot_private_view objects object logicalInside, retained_clear]
  exact Epoch.allocation_epoch bound (GraphLoopsReuse.slotOf objects object) slotInside
    shape element before objects true outputs after

/-- The retained deallocation uses the actual mapped view, frames the caller
heap and leaves the allocator counter unchanged. This is not a failure model.
-/
theorem release_retained {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (bound object next : Nat) (logicalInside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (shape : List Nat) (element : String)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (outputs : List (Runtime Value)) (after : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Epoch.Release [slotRuntime objects (Epoch.view object shape element)]
      (Epoch.clockWorld bound next before) outputs after ↔
      outputs = [] ∧ after = Epoch.clockWorld bound next
        (GraphLoopsReuse.clearMemory objects true object true before) := by
  rw [slot_private_view objects object logicalInside, retained_clear]
  exact Epoch.release_epoch bound next (GraphLoopsReuse.slotOf objects object) slotInside
    shape element before objects true outputs after

/-- The candidate's memory address is obtained from the actual mapped view;
the offset equation preserves every dimension and stride of the canonical view.
-/
theorem slot_address (objects : GraphLoopsReuse.Objects) (object : Nat)
    (shape coordinates : List Nat) (element : String) :
    GraphLoopsShape.address (Epoch.descriptor (GraphLoopsReuse.slotOf objects object) shape element)
      coordinates =
      (GraphLoopsReuse.addressOf objects true object,
        GraphLoopsReuse.Geometry.flatten shape coordinates) :=
  Epoch.address (GraphLoopsReuse.slotOf objects object) shape coordinates element

theorem load_slot_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (object : Nat) (logicalInside : object < objects.length)
    (shape coordinates : List Nat) (element : String)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Load (slotRuntime objects (Epoch.view object shape element) :: coordinates.map Runtime.index)
      before outputs after ↔
      ∃ value, GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
        before.heap (GraphLoopsReuse.addressOf objects true object)
          (GraphLoopsReuse.Geometry.flatten shape coordinates) = some value ∧
        outputs = [.scalar element value] ∧ after = before := by
  rw [slot_private_view objects object logicalInside]
  exact Epoch.load_iff (GraphLoopsReuse.slotOf objects object) shape coordinates element before after outputs

theorem store_slot_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (object : Nat) (logicalInside : object < objects.length)
    (shape coordinates : List Nat) (element : String) (value : Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Store (.scalar element value :: slotRuntime objects (Epoch.view object shape element) ::
      coordinates.map Runtime.index) before outputs after ↔
      GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧ outputs = [] ∧
      after = GraphLoopsReuse.storeMemory objects true object
        (GraphLoopsReuse.Geometry.flatten shape coordinates) value before := by
  rw [slot_private_view objects object logicalInside]
  exact Epoch.store_iff (GraphLoopsReuse.slotOf objects object) shape coordinates element value before after outputs

end Merlin.GraphLoopsMemorySCF.After
