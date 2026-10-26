/- UNCOMPILED. Checked retained-allocation, alias-allocation and free cases.
Ghost registers only justify logical pinning; actual aliases read a live owner.
-/
import Merlin.GraphLoopsMemorySCFAfterRetained

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem allocate_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (operation : String) (item : Epoch.PrivateView)
    (checked : (Epoch.Command.allocate operation item true).Valid plan context sourceClock)
    (slotInside : GraphLoopsReuse.slotOf plan.objects item.object < plan.objects.length)
    (ordinal : clock = GraphLoopsReuse.slotOf plan.objects item.object)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (allocator : meaning.action operation = Epoch.Allocate item.shape item.element) :
    LocalSimulation plan scalar meaning context clock (.keep (.allocate operation item true)) := by
  have itemValid := checked.1.1
  constructor
  · intro before after actual registers callers related run
    have final := (candidate_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar) item.object true before after).mp run
    subst after
    let next := GraphLoopsShape.put actual item.binding.name
      (slotRuntime plan.objects (Epoch.view item.object item.shape item.element))
    have finalRelated := related.define_existing item.binding (Epoch.view item.object item.shape item.element)
      checked.2.2.1.1 (registers.2 item itemValid.1)
    refine ⟨next, (fragments_single meaning (.atom (.action operation [] [item.binding])) actual next _ _).mpr ?_, finalRelated⟩
    apply (scoped_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock item.object
      itemValid.2.1 slotInside ordinal operation item.binding item.shape item.element itemValid.2.2.2.1 itemValid.2.2.2.2
      checked.2.2.1.1 allocator before (registers.2 item itemValid.1) actual related next _).mpr
    exact ⟨_, run, rfl, rfl, finalRelated⟩
  · intro before actual next world registers callers related run
    have first := (fragments_single meaning (.atom (.action operation [] [item.binding])) actual next _ world).mp run
    obtain ⟨after, oldRun, _, finalWorld, finalRelated⟩ :=
      (scoped_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock item.object
        itemValid.2.1 slotInside ordinal operation item.binding item.shape item.element itemValid.2.2.2.1 itemValid.2.2.2.2
        checked.2.2.1.1 allocator before (registers.2 item itemValid.1) actual related next world).mp first
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

theorem release_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (operation : String) (item : Epoch.PrivateView)
    (checked : (Epoch.Command.release operation item true).Valid plan context sourceClock)
    (slotInside : GraphLoopsReuse.slotOf plan.objects item.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (release : meaning.action operation = Epoch.Release) :
    LocalSimulation plan scalar meaning context clock (.keep (.release operation item true)) := by
  have itemValid := checked.1.1
  have bound := checked.2.1 item.binding (List.mem_cons_self)
  constructor
  · intro before after actual registers callers related run
    have final := (candidate_release_iff plan.objects plan.configs (plan.oldMeaning scalar) item.object true before after).mp run
    subst after
    refine ⟨actual, (fragments_single meaning (.atom (.action operation [item.binding] [])) actual actual _ _).mpr ?_, related⟩
    apply (scoped_release_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock item.object
      itemValid.2.1 slotInside operation item.binding bound item.shape item.element release before
      (registers.2 item itemValid.1) actual related actual _).mpr
    exact ⟨_, run, rfl, rfl, related⟩
  · intro before actual next world registers callers related run
    have first := (fragments_single meaning (.atom (.action operation [item.binding] [])) actual next _ world).mp run
    obtain ⟨after, oldRun, _, finalWorld, finalRelated⟩ :=
      (scoped_release_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock item.object
        itemValid.2.1 slotInside operation item.binding bound item.shape item.element release before
        (registers.2 item itemValid.1) actual related next world).mp first
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

theorem alias_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (operation : String) (item owner : Epoch.PrivateView)
    (checked : (Epoch.Command.allocate operation item false).Valid plan context sourceClock)
    (ownerValid : owner.Valid plan) (ownerBound : owner.binding ∈ context)
    (shape : owner.shape = item.shape) (element : owner.element = item.element)
    (sameSlot : GraphLoopsReuse.slotOf plan.objects owner.object = GraphLoopsReuse.slotOf plan.objects item.object)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) :
    LocalSimulation plan scalar meaning context clock (.alias operation item owner) := by
  have itemValid := checked.1.1
  constructor
  · intro before after actual registers callers related run
    have final := (candidate_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar) item.object false before after).mp run
    rw [skipped_clear] at final
    subst after
    let next := GraphLoopsShape.put actual item.binding.name
      (slotRuntime plan.objects (Epoch.view item.object item.shape item.element))
    have ownerRead : before.observed.registers owner.binding.name = Epoch.view owner.object item.shape item.element := by
      simpa only [shape, element] using registers.2 owner ownerValid.1
    have finalRelated := related.define_existing item.binding (Epoch.view item.object item.shape item.element)
      checked.2.2.1.1 (registers.2 item itemValid.1)
    refine ⟨next, (fragments_single meaning (.alias owner.binding item.binding (Epoch.type item.shape item.element))
      actual next _ _).mpr ?_, finalRelated⟩
    exact (scoped_alias_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock owner.object item.object
      ownerValid.2.1 itemValid.2.1 sameSlot owner.binding item.binding ownerBound checked.2.2.1.1 item.shape item.element
      before ownerRead (registers.2 item itemValid.1) actual related next _).mpr ⟨before, run, rfl, rfl, finalRelated⟩
  · intro before actual next world registers callers related run
    have ownerRead : before.observed.registers owner.binding.name = Epoch.view owner.object item.shape item.element := by
      simpa only [shape, element] using registers.2 owner ownerValid.1
    have first := (fragments_single meaning (.alias owner.binding item.binding (Epoch.type item.shape item.element))
      actual next _ world).mp run
    obtain ⟨after, oldRun, _, finalWorld, finalRelated⟩ :=
      (scoped_alias_iff plan.objects plan.configs (plan.oldMeaning scalar) meaning context clock owner.object item.object
        ownerValid.2.1 itemValid.2.1 sameSlot owner.binding item.binding ownerBound checked.2.2.1.1 item.shape item.element
        before ownerRead (registers.2 item itemValid.1) actual related next world).mp first
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

theorem drop_release_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (clock : Nat) (operation : String) (item : Epoch.PrivateView)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) :
    LocalSimulation plan scalar meaning context clock (.dropRelease operation item) := by
  constructor
  · intro before after actual registers callers related run
    have final := (candidate_release_iff plan.objects plan.configs (plan.oldMeaning scalar) item.object false before after).mp run
    rw [skipped_clear] at final
    subst after
    exact ⟨actual, .more ⟨rfl, rfl⟩ (.done _ _), related⟩
  · intro before actual next world registers callers related run
    have first := (fragments_single meaning .empty actual next _ world).mp run
    obtain ⟨rfl, rfl⟩ := first
    exact ⟨before, (candidate_release_iff plan.objects plan.configs (plan.oldMeaning scalar)
      item.object false before before).mpr rfl, rfl, related⟩

end Merlin.GraphLoopsMemorySCF.After
