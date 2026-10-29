/- UNCOMPILED. Scoped allocation, identity aliases and releases. These are the
existing total/private allocator semantics, not a native resource/error model.
The actual allocator counter counts retained allocations, not logical epochs.
-/
import Merlin.GraphLoopsMemorySCFAfterScope

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem scoped_alias_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (context : Context) (clock owner object : Nat)
    (ownerInside : owner < objects.length) (inside : object < objects.length)
    (sameSlot : GraphLoopsReuse.slotOf objects owner = GraphLoopsReuse.slotOf objects object)
    (origin result : Binding) (originBound : origin ∈ context) (fresh : result.name ∉ names context)
    (shape : List Nat) (element : String)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (ownerRead : before.observed.registers origin.name = Epoch.view owner shape element)
    (ghost : before.observed.registers result.name = Epoch.view object shape element)
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Fragment.alias origin result (Epoch.type shape element)).step meaning actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.allocate object false) before after ∧
        next = GraphLoopsShape.put actual result.name (slotRuntime objects (Epoch.view object shape element)) ∧
        world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects (result :: context) after.observed.registers next := by
  have read : actual origin.name = slotRuntime objects (Epoch.view object shape element) := by
    rw [related.private_view origin originBound owner ownerInside shape element ownerRead,
      slot_private_view objects object inside, sameSlot]
  have kind : (slotRuntime objects (Epoch.view object shape element) : Runtime Value).kind =
      .view (Epoch.type shape element) := slot_kind objects _
  rw [alias_actual_iff meaning origin result (Epoch.type shape element) actual _ read kind]
  simp only [candidate_allocate_iff, skipped_clear, exists_eq_left]
  constructor
  · rintro ⟨rfl, rfl⟩
    exact ⟨rfl, rfl, related.define_existing result (Epoch.view object shape element) fresh ghost⟩
  · rintro ⟨assigned, world, _⟩
    exact ⟨assigned, world⟩

theorem scoped_allocate_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (context : Context) (clock object : Nat) (inside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < objects.length)
    (ordinal : clock = GraphLoopsReuse.slotOf objects object)
    (operation : String) (result : Binding) (shape : List Nat) (element : String)
    (kind : result.kind = .view (Epoch.type shape element)) (plain : result.known = none)
    (fresh : result.name ∉ names context)
    (allocator : meaning.action operation = Epoch.Allocate shape element)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (ghost : before.observed.registers result.name = Epoch.view object shape element)
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation [] [result]).step meaning actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.allocate object true) before after ∧
        next = GraphLoopsShape.put actual result.name (slotRuntime objects (Epoch.view object shape element)) ∧
        world = Epoch.clockWorld objects.length (clock + 1) after ∧
        RegistersRelated objects (result :: context) after.observed.registers next := by
  simp only [candidate_allocate_iff, exists_eq_left, Epoch.Atom.step, readArguments, List.map_nil, allocator]
  simp only [allocate_retained objects objects.length object clock inside slotInside ordinal shape element before]
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, assigned⟩
    refine ⟨assigned, world, ?_⟩
    rw [assigned]
    exact related.define_existing result (Epoch.view object shape element) fresh ghost
  · rintro ⟨assigned, world, _⟩
    exact ⟨[slotRuntime objects (Epoch.view object shape element)], ⟨rfl, world⟩,
      .cons (slot_holds objects result _ (Binding.holds_plain result _ kind.symm plain)) .nil, assigned⟩

theorem scoped_release_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (context : Context) (clock object : Nat) (inside : object < objects.length)
    (slotInside : GraphLoopsReuse.slotOf objects object < objects.length)
    (operation : String) (buffer : Binding) (bound : buffer ∈ context) (shape : List Nat) (element : String)
    (release : meaning.action operation = Epoch.Release)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (pinned : before.observed.registers buffer.name = Epoch.view object shape element)
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    (Epoch.Atom.action operation [buffer] []).step meaning actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.release object true) before after ∧
        next = actual ∧ world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  have read : actual buffer.name = slotRuntime objects (Epoch.view object shape element) := by
    rw [related buffer bound, pinned]
  simp only [candidate_release_iff, exists_eq_left, Epoch.Atom.step, readArguments, List.map_cons,
    List.map_nil, Bool.false_eq_true, ↓reduceIte, read, release]
  rw [release_retained objects objects.length object clock inside slotInside shape element before]
  constructor
  · rintro ⟨values, ⟨rfl, world⟩, _, assigned⟩
    refine ⟨assigned, world, ?_⟩
    rw [assigned]
    exact related
  · rintro ⟨assigned, world, _⟩
    exact ⟨[], ⟨rfl, world⟩, .nil, assigned⟩

theorem scoped_drop_release_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (old : GraphLoopsMemorySCF.Interpretation Value (Epoch.PublicState Value User))
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (context : Context) (clock object : Nat)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Fragment.empty.step meaning actual (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs old true true (.release object false) before after ∧
        next = actual ∧ world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  simp only [Fragment.step, candidate_release_iff, skipped_clear, exists_eq_left]
  constructor
  · rintro ⟨rfl, rfl⟩; exact ⟨rfl, rfl, related⟩
  · rintro ⟨assigned, world, _⟩; exact ⟨assigned, world⟩

end Merlin.GraphLoopsMemorySCF.After
