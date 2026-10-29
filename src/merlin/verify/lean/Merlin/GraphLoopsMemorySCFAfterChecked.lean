/- UNCOMPILED. Exhaustive finite expansion → local operational simulation.
The eight source Command constructors and all four Expansion constructors are
discharged here. No whole-run or local-simulation premise is accepted by the
final bound_local_simulations statement. This remains a candidate until compiled.
-/
import Merlin.GraphLoopsMemorySCFAfterAccessLocal
import Merlin.GraphLoopsMemorySCFAfterAllocationLocal
import Merlin.GraphLoopsMemorySCFAfterUntiledLocal
import Merlin.GraphLoopsMemorySCFAfterLineLocal
import Merlin.GraphLoopsMemorySCFAfterLineTiledLocal

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem Expansion.local_checked {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (target : List Expansion)
    (noOverrides : NoOpcodeOverrides plan indices)
    (slots : ∀ object ∈ List.range plan.objects.length, GraphLoopsReuse.slotOf plan.objects object < plan.objects.length)
    (context : Context) (sourceClock clock : Nat) (expansion : Expansion)
    (checked : expansion.source.Valid plan context sourceClock)
    (facts : expansion.Facts bits plan indices context clock)
    (observer : ObserverBinding (expandedModule plan kernel target) expansion.source)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    LocalSimulation plan scalar (expandedMeaning (User := User) bits indices plan kernel target scalar depth)
      context clock expansion := by
  have inside (object : Nat) (bound : object < plan.objects.length) := slots object (List.mem_range.mpr bound)
  cases expansion with
  | keep command =>
    cases command with
    | constant key result value => exact constant_local plan context sourceClock clock key result value checked scalar _
    | binary key operation left right result element =>
      exact binary_local bits indices plan kernel target noOverrides context sourceClock clock key operation
        left right result element checked scalar depth
    | allocate operation item retained =>
      cases retained with
      | false => exact False.elim facts
      | true =>
        rcases checked.1 with ⟨itemValid, _, opcode, notCall⟩
        have action := (expanded_primitive (User := User) bits indices plan kernel target noOverrides scalar depth operation
          (.allocate item.shape item.element) opcode notCall).trans
            (Epoch.graph_allocate plan.opcodes scalar operation item.shape item.element opcode)
        exact allocate_local plan context sourceClock clock operation item checked (inside item.object itemValid.2.1)
          facts.2.1.symm scalar _ action
    | release operation item retained =>
      cases retained with
      | false => exact False.elim facts
      | true =>
        rcases checked.1 with ⟨itemValid, opcode, notCall⟩
        have action := (expanded_primitive (User := User) bits indices plan kernel target noOverrides scalar depth operation .release
          opcode notCall).trans (Epoch.graph_release plan.opcodes scalar operation opcode)
        exact release_local plan context sourceClock clock operation item checked (inside item.object itemValid.2.1) scalar _ action
    | «initialize» link => exact False.elim facts
    | initializeLine link => exact False.elim facts
    | load key operation access result =>
      rcases checked.1 with ⟨_, accessValid, opcode, notCall, _⟩
      have action := (expanded_primitive (User := User) bits indices plan kernel target noOverrides scalar depth operation .load
        opcode notCall).trans (Epoch.graph_load plan.opcodes scalar operation opcode)
      exact load_local plan context sourceClock clock key operation access result checked
        (inside access.object accessValid.1.2.1) scalar _ action
    | store key operation access operand =>
      rcases checked.1 with ⟨_, accessValid, opcode, notCall, _⟩
      have action := (expanded_primitive (User := User) bits indices plan kernel target noOverrides scalar depth operation .store
        opcode notCall).trans (Epoch.graph_store plan.opcodes scalar operation opcode)
      exact store_local plan context sourceClock clock key operation access operand checked
        (inside access.object accessValid.1.2.1) scalar _ action
    | observer key operation name input buffer result program =>
      exact observer_local bits indices plan kernel target noOverrides context sourceClock clock key operation name
        input buffer result program checked observer scalar depth
  | alias operation item owner =>
    rcases facts with ⟨_, ownerValid, ownerBound, shape, element, slotsEqual, _⟩
    exact alias_local plan context sourceClock clock operation item owner checked ownerValid ownerBound
      shape.symm element.symm slotsEqual.symm scalar _
  | dropRelease operation item => exact drop_release_local plan context clock operation item scalar _
  | untiled link =>
    rcases facts with ⟨_,positive⟩
    rcases checked.1 with ⟨_,_,_,_,objectInside,storeOpcode,loopOpcode⟩
    let base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
      withCalls (withIndex (Epoch.scalarAllocationBase plan.opcodes scalar) bits indices)
        (expandedModule plan kernel target)
        (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
          (expandedModule plan kernel target) false (depth+1))
    have memory := expanded_memory (User := User) bits indices plan planValid kernel target scalar depth
    have result := untiled_initializer_local plan planValid context sourceClock clock link checked positive
      (inside link.config.object objectInside) scalar base (Epoch.memoryOpcode plan.opcodes)
      (by simp only [Epoch.memoryOpcode,storeOpcode]) (by simp only [Epoch.memoryOpcode,loopOpcode])
    rw [← memory] at result
    exact result
  | untiledLine link =>
    rcases facts with ⟨_,positive⟩
    rcases checked.1 with ⟨_,_,_,_,objectInside,storeOpcode,loopOpcode⟩
    let base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
      withCalls (withIndex (Epoch.scalarAllocationBase plan.opcodes scalar) bits indices)
        (expandedModule plan kernel target)
        (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
          (expandedModule plan kernel target) false (depth+1))
    have memory := expanded_memory (User := User) bits indices plan planValid kernel target scalar depth
    have result := line_initializer_local plan planValid context sourceClock clock link checked positive
      (inside link.config.object objectInside) scalar base (Epoch.memoryOpcode plan.opcodes)
      (by simp only [Epoch.memoryOpcode,storeOpcode]) (by simp only [Epoch.memoryOpcode,loopOpcode])
    rw [← memory] at result
    exact result
  | tiled link program =>
    rcases facts with ⟨_, sameUniform, _, _, controls, bounds, fresh,
      addIndex, minIndex, addOld, minOld, addNotCall, minNotCall⟩
    rcases checked.1 with ⟨_, _, _, _, objectInside, storeOpcode, loopOpcode⟩
    let base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
      withCalls (withIndex (Epoch.scalarAllocationBase plan.opcodes scalar) bits indices)
        (expandedModule plan kernel target)
        (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
          (expandedModule plan kernel target) false (depth + 1))
    have memory := expanded_memory (User := User) bits indices plan planValid kernel target scalar depth
    have add := expanded_generated (User := User) bits indices plan kernel target scalar depth program.addKey .add addIndex addOld addNotCall
    have minimum := expanded_generated (User := User) bits indices plan kernel target scalar depth program.minKey .minimum minIndex minOld minNotCall
    rw [memory] at add minimum
    have result := initializer_local plan planValid context sourceClock clock link program checked sameUniform controls bits bounds
      (fresh program.tile (List.mem_cons_self)) (inside link.config.object objectInside) scalar base
      (Epoch.memoryOpcode plan.opcodes)
      (by simp only [sameUniform, Epoch.memoryOpcode, storeOpcode])
      (by simp only [sameUniform, Epoch.memoryOpcode, loopOpcode]) add minimum
    rw [← memory] at result
    exact result
  | tiledLine link program =>
    rcases facts with ⟨_, sameUniform, _, _, controls, bounds, fresh,
      addIndex, minIndex, addOld, minOld, addNotCall, minNotCall⟩
    rcases checked.1 with ⟨_, _, _, _, objectInside, storeOpcode, loopOpcode⟩
    let base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
      withCalls (withIndex (Epoch.scalarAllocationBase plan.opcodes scalar) bits indices)
        (expandedModule plan kernel target)
        (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
          (expandedModule plan kernel target) false (depth + 1))
    have memory := expanded_memory (User := User) bits indices plan planValid kernel target scalar depth
    have add := expanded_generated (User := User) bits indices plan kernel target scalar depth program.addKey .add addIndex addOld addNotCall
    have minimum := expanded_generated (User := User) bits indices plan kernel target scalar depth program.minKey .minimum minIndex minOld minNotCall
    rw [memory] at add minimum
    have result := tiled_line_initializer_local plan planValid context sourceClock clock link program checked sameUniform controls bits bounds
      (fresh program.tile (List.mem_cons_self)) (inside link.config.object objectInside) scalar base
      (Epoch.memoryOpcode plan.opcodes)
      (by simp only [sameUniform, Epoch.memoryOpcode, storeOpcode])
      (by simp only [sameUniform, Epoch.memoryOpcode, loopOpcode]) add minimum
    rw [← memory] at result
    exact result

theorem checked_local_list {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (target : List Expansion)
    (noOverrides : NoOpcodeOverrides plan indices)
    (slots : ∀ object ∈ List.range plan.objects.length, GraphLoopsReuse.slotOf plan.objects object < plan.objects.length)
    (context : Context) (sourceClock clock : Nat) (expansions : List Expansion)
    (checked : Epoch.CommandsValid plan context sourceClock (expansions.map Expansion.source))
    (facts : ExpansionsValid bits plan indices context clock expansions)
    (observers : ∀ expansion ∈ expansions, ObserverBinding (expandedModule plan kernel target) expansion.source)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    LocalSimulations plan scalar (expandedMeaning (User := User) bits indices plan kernel target scalar depth)
      context clock expansions := by
  induction expansions generalizing context sourceClock clock with
  | nil => trivial
  | cons expansion rest induction =>
    exact ⟨expansion.local_checked bits indices plan planValid kernel target noOverrides slots context sourceClock clock
      checked.1 facts.1 (observers expansion List.mem_cons_self) scalar depth,
      induction (extendContext context expansion.source.results) (expansion.source.nextClock sourceClock)
        (expansion.nextClock clock) checked.2 facts.2
        (fun item member => observers item (List.mem_cons_of_mem _ member))⟩

/-- The LOCAL simulation used by ordered-list/return lifting is derived from
the checked finite source plan and expanded syntax, including actual call bodies.
No simulation, full-environment equality or complete-run equality is a premise.
-/
theorem bound_local_simulations {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (kernelFacts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions) (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    LocalSimulations plan scalar (expandedMeaning (User := User) bits indices plan kernel expansions scalar depth)
      kernel.function.parameters 0 expansions := by
  refine checked_local_list bits indices plan planValid kernel expansions binding.2.2.1 boundary.1
    kernel.function.parameters 0 0 expansions ?_ binding.2.1 ?_ scalar depth
  · rw [binding.1]; exact kernelFacts.2.2.1
  · intro expansion member
    have sourceMember : expansion.source ∈ kernel.commands := by
      rw [← binding.1]
      exact List.mem_map_of_mem member
    exact boundary.2.2.2.1 expansion.source sourceMember

end Merlin.GraphLoopsMemorySCF.After
