/- The existing ordered-state semantics and allocation geometry are joined at
the complete actual counted-loop boundary. Adapter effect truth, disjoint
external backing identities and initial external bytes remain explicit premises.
There is no allocation-changing optimization or renderer/hardware claim here. -/
import Merlin.GraphLoopsStorage

set_option autoImplicit false

namespace Merlin.GraphLoops.Bound

open Storage

structure Allocation where
  region : Region
  minimum : Nat
  capacity : Nat
  alignment : Nat
  deriving DecidableEq, Repr

def Allocation.Valid (allocation : Allocation) : Prop :=
  allocation.minimum ≤ allocation.region.address ∧ 0 < allocation.region.extent ∧
  ResourceGeometry.Fits allocation.region.address allocation.region.extent allocation.capacity allocation.alignment

instance (allocation : Allocation) : Decidable allocation.Valid :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

structure Event where
  call : ClosedCall
  effect : Effect
  deriving DecidableEq, Repr

def check (loop : Counted) (actual : List ClosedCall) (allocations : List Allocation)
    (external initialized : List Region) (events : List (List Event)) : Bool :=
  GraphLoops.check loop actual &&
    match expand loop with
    | none => false
    | some calls =>
      decide (calls = events.map (List.map Event.call) ∧
        (∀ allocation ∈ allocations, allocation.Valid) ∧
        (∀ known ∈ initialized, Covered external known) ∧
        (∀ outside ∈ external, ∀ allocation ∈ allocations, Separated outside allocation.region)) &&
      checkPeriods (external ++ allocations.map Allocation.region) initialized
        (allocations.map Allocation.region) loop.extent loop.body.length (events.map (List.map Event.effect))

theorem check_sound {loop : Counted} {actual : List ClosedCall} {allocations : List Allocation}
    {external initialized : List Region} {events : List (List Event)}
    (accepted : check loop actual allocations external initialized events = true) :
    GraphLoops.check loop actual = true ∧
    expand loop = some (events.map (List.map Event.call)) ∧
    (∀ allocation ∈ allocations, allocation.Valid) ∧
    (∀ known ∈ initialized, Covered external known) ∧
    (∀ outside ∈ external, ∀ allocation ∈ allocations, Separated outside allocation.region) ∧
    checkPeriods (external ++ allocations.map Allocation.region) initialized
      (allocations.map Allocation.region) loop.extent loop.body.length (events.map (List.map Event.effect)) = true := by
  simp only [check, Bool.and_eq_true] at accepted
  obtain ⟨relation, storage⟩ := accepted
  cases expanded : expand loop with
  | none => simp [expanded] at storage
  | some calls =>
    simp only [expanded, Bool.and_eq_true, decide_eq_true_eq] at storage
    obtain ⟨⟨same, valid, inputs, separate⟩, safe⟩ := storage
    exact ⟨relation, congrArg some same, valid, inputs, separate, safe⟩

/-- Exact complete source/result correspondence AND every period's admission;
the two are attached to the same ordered actual event rows. -/
theorem complete_boundary {loop : Counted} {actual : List ClosedCall} {allocations : List Allocation}
    {external initialized : List Region} {events : List (List Event)}
    (accepted : check loop actual allocations external initialized events = true) :
    actual = (events.map (List.map Event.call)).flatten ∧
    actual.length = loop.extent * loop.body.length ∧
    events.length = loop.extent ∧
    ∀ block ∈ events, block.length = loop.body.length ∧
      ∃ final, Trace (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
        ⟨initialized, [], []⟩ (block.map Event.effect) final ∧
        final.pending = [] ∧ ∀ object ∈ allocations.map Allocation.region, Covered final.initialized object := by
  have facts := check_sound accepted
  obtain ⟨calls, expanded, same⟩ := GraphLoops.check_sound facts.1
  have equal := Option.some.inj (expanded.symm.trans facts.2.1)
  subst calls
  have safe := periods_sound facts.2.2.2.2.2
  refine ⟨same.symm, complete_call_count facts.1, ?_, ?_⟩
  · simpa only [List.length_map] using safe.1
  · intro block member
    have this := safe.2 (block.map Event.effect) (List.mem_map.mpr ⟨block, member, rfl⟩)
    exact ⟨by simpa only [List.length_map] using this.1, this.2⟩

/-- No independent purity or commutativity law is needed: order is identical.
The handler is an explicit deterministic total state semantics, not an inferred
claim that a target instruction or an undefined arithmetic operation is total. -/
theorem preserves {loop : Counted} {actual : List ClosedCall} {allocations : List Allocation}
    {external initialized : List Region} {events : List (List Event)}
    (accepted : check loop actual allocations external initialized events = true) {Value : Type}
    (run : ClosedCall → StateEffects.State Value → StateEffects.State Value) (initial : StateEffects.State Value) :
    (events.map (List.map Event.call)).foldl
        (fun state block => StateEffects.execute (block.map run) state) initial =
      StateEffects.execute (actual.map run) initial := by
  rw [(complete_boundary accepted).1]
  exact ordered_execution _ run initial

theorem allocation_point {allocation : Allocation} (valid : allocation.Valid) {point : Nat}
    (inside : allocation.region.address ≤ point ∧ point < allocation.region.address + allocation.region.extent) :
    allocation.minimum ≤ point ∧ point < allocation.capacity := by
  exact ⟨Nat.le_trans valid.1 inside.1,
    ResourceGeometry.fitsContains _ _ _ _ _ valid.2.2 inside⟩

end Merlin.GraphLoops.Bound
