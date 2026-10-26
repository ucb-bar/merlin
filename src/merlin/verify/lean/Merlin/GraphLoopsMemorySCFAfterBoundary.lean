/- UNCOMPILED. Full block return/caller observation and actual entry registers.
The local-simulation obligation remains explicit: these lifting lemmas do not
yet certify that the finite expansion plan supplies every local proof/call edge.
-/
import Merlin.GraphLoopsMemorySCFAfterExecution
import Merlin.GraphLoopsShapeEpochOutcomes

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def EntrySlotFacts (plan : Epoch.PlanContext) (function : Function) : Prop :=
  ∀ binding ∈ function.parameters, NoViewKind binding.kind ∨
    ∃ caller ∈ plan.callers, caller.binding = binding

instance (plan : Epoch.PlanContext) (function : Function) : Decidable (EntrySlotFacts plan function) := by
  unfold EntrySlotFacts
  infer_instance

/-- Logical allocation ghosts are absent from the actual input environment.
They are related only after the corresponding allocation/alias defines an SSA
name. Actual view parameters must be caller views outside the private partition.
-/
theorem entry_related {Value User : Type} (plan : Epoch.PlanContext) (function : Function)
    (facts : Epoch.EntryFacts plan function) (domain : EntrySlotFacts plan function)
    (inputs : List (Runtime Value)) (typed : ValuesHold function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv function.parameters inputs))
    (heap : Heap Value) (user : User) :
    RegistersRelated plan.objects function.parameters
      (Epoch.entryMemory plan function inputs heap user).observed.registers
      (assign emptyEnv function.parameters inputs) := by
  intro binding member
  have outside : binding.name ∉ plan.views.map (fun item => item.binding.name) := by
    intro mapped
    obtain ⟨item, itemMember, equal⟩ := List.mem_map.mp mapped
    exact facts.2.2.1 item itemMember (equal ▸ List.mem_map_of_mem member)
  change _ = slotRuntime plan.objects (Epoch.seedViews _ plan.views binding.name)
  rw [Epoch.seed_other _ plan.views binding.name outside]
  rcases domain binding member with scalar | ⟨caller, callerMember, rfl⟩
  · have valid := (Epoch.entry_registers plan function facts inputs typed heap user).1 binding member
    change binding.Holds (Epoch.seedViews _ plan.views binding.name) at valid
    rw [Epoch.seed_other _ plan.views binding.name outside] at valid
    exact (scalar_slot_fixed plan.objects binding _ valid scalar).symm
  · obtain ⟨object, external, read⟩ := callers caller callerMember
    rw [read, slot_caller_view plan.objects object external]

theorem candidate_returned_iff {Value User : Type} (plan : Epoch.PlanContext)
    (scalar : Epoch.BinaryMeaning Value) (key : String) (returned : List Binding)
    (selected : Epoch.specAt plan.specifications key = some (.returned returned))
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) :
    GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true (.retained key) before after ↔
      after = Epoch.returnMemory returned before := by
  rw [retained_modes plan.objects plan.configs plan.objects.length plan.specifications plan.mapping
    scalar key (.returned returned) selected]
  exact Epoch.returned_step_iff plan.objects.length plan.specifications scalar plan.mapping key returned
    selected plan.objects plan.configs before after

theorem expanded_block {Value User : Type} (plan : Epoch.PlanContext)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (expansions : List Expansion) (context : Context) (sourceClock clock : Nat)
    (checked : Epoch.CommandsValid plan context sourceClock (expansions.map Expansion.source))
    (local : LocalSimulations plan scalar meaning context clock expansions)
    (returnKey : String) (returned : List Binding)
    (returnSelected : Epoch.specAt plan.specifications returnKey = some (.returned returned))
    (returnBound : AllBound (Epoch.finalContext context (expansions.map Expansion.source)) returned)
    (returnDomain : NoViewBindings returned)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value)
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (callers : Epoch.CallersValid plan.objects.length plan.callers before.observed.registers)
    (related : RegistersRelated plan.objects context before.observed.registers actual)
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Eval meaning false [] (fragmentsCode (expansions.flatMap Expansion.fragments) returned) actual
      (Epoch.clockWorld plan.objects.length clock before) outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
        ((expansions.map Expansion.source).map Epoch.Command.instruction ++ [.retained returnKey]) before after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (physicalFinalClock clock expansions) after := by
  rw [fragments_complete]
  constructor
  · rintro ⟨next, run, outputsEqual⟩
    obtain ⟨middle, oldRun, worldEqual, finalRelated, finalRegisters, _⟩ :=
      expansions_backward plan scalar meaning expansions context sourceClock clock checked local before actual next world
        registers callers related run
    refine ⟨Epoch.returnMemory returned middle, ?_, ?_, worldEqual⟩
    · apply (Epoch.execution_append _ _ _ _ _ _ _ _ _).mpr
      refine ⟨middle, oldRun, (Epoch.execution_single _ _ _ _ _ _ _ _).mpr ?_⟩
      exact (candidate_returned_iff plan scalar returnKey returned returnSelected middle _).mpr rfl
    · exact outputsEqual.trans (finalRelated.returns finalRegisters.1 returned returnBound returnDomain)
  · rintro ⟨after, run, outputsEqual, worldEqual⟩
    obtain ⟨middle, initialRun, returnRun⟩ := (Epoch.execution_append _ _ _ _ _ _ _ _ _).mp run
    have returnedEqual := (candidate_returned_iff plan scalar returnKey returned returnSelected middle after).mp
      ((Epoch.execution_single _ _ _ _ _ _ _ _).mp returnRun)
    subst after
    obtain ⟨next, expandedRun, finalRelated, finalRegisters, _⟩ :=
      expansions_forward plan scalar meaning expansions context sourceClock clock checked local before middle actual
        registers callers related initialRun
    refine ⟨next, ?_, ?_⟩
    · rw [worldEqual]; exact expandedRun
    · exact outputsEqual.trans (finalRelated.returns finalRegisters.1 returned returnBound returnDomain).symm

/-- Exactly the complete caller heap, returned values and user state survive
the frame. Allocation counter/private cells are deliberately unobservable.
-/
theorem expanded_observation {Value User : Type} (bound clock : Nat)
    (memory : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (Epoch.AllocationPublic User))
    (returned : outputs = memory.observed.returned) (framed : world = Epoch.clockWorld bound clock memory) :
    Epoch.observeWorld bound outputs world = Epoch.observePublic bound memory.observed := by
  rw [returned, framed]
  exact Epoch.observe_clock bound clock memory

end Merlin.GraphLoopsMemorySCF.After
