/- UNCOMPILED. Discharge the checked load/store constructors of LocalSimulation.
No local semantic correspondence is left as a premise: only selected concrete
operation meaning, checked bindings and the finite physical-slot bound remain.
-/
import Merlin.GraphLoopsMemorySCFAfterRetained

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem load_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (key operation : String) (access : Epoch.Access) (result : Binding)
    (checked : (Epoch.Command.load key operation access result).Valid plan context sourceClock)
    (slotInside : GraphLoopsReuse.slotOf plan.objects access.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (load : meaning.action operation = Load) :
    LocalSimulation plan scalar meaning context clock (.keep (.load key operation access result)) := by
  have accessValid := checked.1.2.1
  have bufferBound := checked.2.1 access.buffer (List.mem_cons_self)
  have indicesBound : AllBound context access.indices :=
    fun binding member => checked.2.1 binding (List.mem_cons_of_mem _ member)
  constructor
  · intro before after actual registers callers related run
    obtain ⟨coordinates, value, reads, read, holds, rfl⟩ :=
      (candidate_load_iff plan.objects.length plan.specifications scalar plan.mapping key access result checked.1.1
        plan.objects plan.configs accessValid.1.2.2.1 before after).mp run
    let next := GraphLoopsShape.put actual result.name (.scalar access.element value)
    have finalRelated := related.define result (.scalar access.element value) checked.2.2.1.1
    refine ⟨next, (fragments_single meaning (.atom (.action operation (access.buffer :: access.indices) [result]))
      actual next _ _).mpr ?_, finalRelated⟩
    apply (scoped_load_iff plan.objects plan.configs plan.specifications plan.mapping scalar context clock key operation
      access result checked.1.1 accessValid.1.2.1 slotInside accessValid.1.2.2.1 bufferBound indicesBound accessValid.2
      checked.2.2.1.1 meaning load before (registers.2 _ accessValid.1.1) registers.1 actual related next _).mpr
    exact ⟨_, value, run, rfl, rfl, rfl, finalRelated⟩
  · intro before actual next world registers callers related run
    have first := (fragments_single meaning (.atom (.action operation (access.buffer :: access.indices) [result]))
      actual next _ world).mp run
    obtain ⟨after, value, oldRun, _, _, finalWorld, finalRelated⟩ :=
      (scoped_load_iff plan.objects plan.configs plan.specifications plan.mapping scalar context clock key operation
        access result checked.1.1 accessValid.1.2.1 slotInside accessValid.1.2.2.1 bufferBound indicesBound accessValid.2
        checked.2.2.1.1 meaning load before (registers.2 _ accessValid.1.1) registers.1 actual related next world).mp first
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

theorem store_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (key operation : String) (access : Epoch.Access) (operand : Binding)
    (checked : (Epoch.Command.store key operation access operand).Valid plan context sourceClock)
    (slotInside : GraphLoopsReuse.slotOf plan.objects access.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (store : meaning.action operation = Store) :
    LocalSimulation plan scalar meaning context clock (.keep (.store key operation access operand)) := by
  have accessValid := checked.1.2.1
  have operandScalar := checked.1.2.2.2.2
  have operandBound := checked.2.1 operand (List.mem_cons_self)
  have bufferBound := checked.2.1 access.buffer (List.mem_cons_of_mem _ List.mem_cons_self)
  have indicesBound : AllBound context access.indices :=
    fun binding member => checked.2.1 binding (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))
  constructor
  · intro before after actual registers callers related run
    obtain ⟨coordinates, value, reads, operandRead, rfl⟩ :=
      (candidate_store_iff plan.objects.length plan.specifications scalar plan.mapping key access operand checked.1.1
        plan.objects plan.configs accessValid.1.2.2.1 before after).mp run
    refine ⟨actual, (fragments_single meaning (.atom (.action operation (operand :: access.buffer :: access.indices) []))
      actual actual _ _).mpr ?_, related⟩
    apply (scoped_store_iff plan.objects plan.configs plan.specifications plan.mapping scalar context clock key operation
      access operand checked.1.1 accessValid.1.2.1 slotInside accessValid.1.2.2.1 bufferBound operandBound indicesBound
      accessValid.2 meaning store before (registers.2 _ accessValid.1.1) value operandRead registers.1 actual related actual _).mpr
    exact ⟨_, run, rfl, rfl, related⟩
  · intro before actual next world registers callers related run
    obtain ⟨value, operandRead⟩ := Epoch.scalar_of_kind (before.observed.registers operand.name) access.element
      ((registers.1 operand operandBound).1.trans operandScalar.1)
    have first := (fragments_single meaning (.atom (.action operation (operand :: access.buffer :: access.indices) []))
      actual next _ world).mp run
    obtain ⟨after, oldRun, _, finalWorld, finalRelated⟩ :=
      (scoped_store_iff plan.objects plan.configs plan.specifications plan.mapping scalar context clock key operation
        access operand checked.1.1 accessValid.1.2.1 slotInside accessValid.1.2.2.1 bufferBound operandBound indicesBound
        accessValid.2 meaning store before (registers.2 _ accessValid.1.1) value operandRead registers.1 actual related next world).mp first
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

end Merlin.GraphLoopsMemorySCF.After
