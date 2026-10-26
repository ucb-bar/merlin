/- UNCOMPILED. Ordinary expanded-target allocation/alias/release fragments
correspond to the preserved candidate epoch instructions under slotEnv and the
actual retained-allocation counter. Allocation failure stays outside this model.
-/
import Merlin.GraphLoopsMemorySCFAfterFragments
import Merlin.GraphLoopsShapeEpochAllocationSteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem candidate_allocate_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (object : Nat) (retained : Bool) (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning true true (.allocate object retained) before after ↔
      after = GraphLoopsReuse.clearMemory objects true object retained before := by
  constructor
  · intro run; cases run with
    | ordinary _ run => cases run; rfl
  · rintro rfl
    exact .ordinary (by trivial) (.allocate object retained before)

theorem candidate_release_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (object : Nat) (retained : Bool) (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning true true (.release object retained) before after ↔
      after = GraphLoopsReuse.clearMemory objects true object retained before := by
  constructor
  · intro run; cases run with
    | ordinary _ run => cases run; rfl
  · rintro rfl
    exact .ordinary (by trivial) (.release object retained before)

theorem alias_allocation_fragment {Value User : Type}
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (bound clock owner object : Nat) (ownerInside : owner < objects.length) (objectInside : object < objects.length)
    (sameSlot : GraphLoopsReuse.slotOf objects owner = GraphLoopsReuse.slotOf objects object)
    (origin result : Binding) (shape : List Nat) (element : String)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (ownerValue : before.observed.registers origin.name = Epoch.view owner shape element)
    (objectValue : before.observed.registers result.name = Epoch.view object shape element)
    (env : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Fragment.alias origin result (Epoch.type shape element)).step meaning
      (slotEnv objects before.observed.registers) (Epoch.clockWorld bound clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.allocate object false) before after ∧
        env = slotEnv objects after.observed.registers ∧ world = Epoch.clockWorld bound clock after := by
  rw [alias_slot meaning objects owner object ownerInside objectInside sameSlot origin result shape element
    before.observed.registers ownerValue]
  rw [Epoch.put_existing _ _ _ objectValue]
  simp only [candidate_allocate_iff, skipped_clear, exists_eq_left]

theorem skipped_release_fragment {Value User : Type}
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (bound clock object : Nat) (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (env : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Fragment.empty.step meaning (slotEnv objects before.observed.registers)
      (Epoch.clockWorld bound clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.release object false) before after ∧
        env = slotEnv objects after.observed.registers ∧ world = Epoch.clockWorld bound clock after := by
  simp only [Fragment.step, candidate_release_iff, skipped_clear, exists_eq_left]

theorem retained_allocation_fragment {Value User : Type}
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (bound clock object : Nat) (inside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (ordinal : clock = GraphLoopsReuse.slotOf objects object)
    (key : String) (result : Binding) (shape : List Nat) (element : String)
    (kind : result.kind = .view (Epoch.type shape element)) (plain : result.known = none)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (allocator : meaning.action key = Epoch.Allocate shape element)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (pinned : before.observed.registers result.name = Epoch.view object shape element)
    (env : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Fragment.atom (.action key [] [result])).step meaning (slotEnv objects before.observed.registers)
      (Epoch.clockWorld bound clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.allocate object true) before after ∧
        env = slotEnv objects after.observed.registers ∧ world = Epoch.clockWorld bound (clock + 1) after := by
  simp only [candidate_allocate_iff, exists_eq_left]
  change (∃ values, meaning.action key [] (Epoch.clockWorld bound clock before) values world ∧
    ValuesHold [result] values ∧ env = assign (slotEnv objects before.observed.registers) [result] values) ↔ _
  rw [allocator]
  simp only [allocate_retained objects bound object clock inside slotInside ordinal shape element before]
  have existing : GraphLoopsShape.put (slotEnv objects before.observed.registers) result.name
      (slotRuntime objects (Epoch.view object shape element)) = slotEnv objects before.observed.registers := by
    rw [← slot_put, Epoch.put_existing _ _ _ pinned]
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, assigned⟩
    exact ⟨by simpa only [assign, existing, GraphLoopsReuse.clearMemory] using assigned, world⟩
  · rintro ⟨assigned, world⟩
    refine ⟨[slotRuntime objects (Epoch.view object shape element)], ⟨rfl, world⟩,
      .cons (slot_holds objects result _ (Binding.holds_plain result _ kind.symm plain)) .nil, ?_⟩
    simpa only [assign, existing, GraphLoopsReuse.clearMemory] using assigned

theorem retained_release_fragment {Value User : Type}
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (bound clock object : Nat) (inside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < bound)
    (key : String) (buffer : Binding) (shape : List Nat) (element : String)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (release : meaning.action key = Epoch.Release)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (pinned : before.observed.registers buffer.name = Epoch.view object shape element)
    (env : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Fragment.atom (.action key [buffer] [])).step meaning (slotEnv objects before.observed.registers)
      (Epoch.clockWorld bound clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.release object true) before after ∧
        env = slotEnv objects after.observed.registers ∧ world = Epoch.clockWorld bound clock after := by
  simp only [candidate_release_iff, exists_eq_left]
  change (∃ values, meaning.action key [slotRuntime objects (before.observed.registers buffer.name)]
    (Epoch.clockWorld bound clock before) values world ∧ ValuesHold [] values ∧
    env = assign (slotEnv objects before.observed.registers) [] values) ↔ _
  rw [release, pinned]
  simp only [release_retained objects bound object clock inside slotInside shape element before]
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, assigned⟩
    exact ⟨assigned, world⟩
  · rintro ⟨assigned, world⟩
    exact ⟨[], ⟨rfl, world⟩, .nil, assigned⟩

end Merlin.GraphLoopsMemorySCF.After
