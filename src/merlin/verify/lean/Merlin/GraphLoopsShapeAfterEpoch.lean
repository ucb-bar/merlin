/- Candidate actual-middle adapter. UNCOMPILED under the native runtime hold.
This closes only the normalized shape boundary: the raw independently decoded
middle is evaluated ordinarily, and its bijective renaming is checked against
the existing per-operation epoch plan. The final reuse/segmentation program
still needs its own independently decoded operational bridge. -/
import Merlin.GraphLoopsShapeAfter
import Merlin.GraphLoopsShapeEpochChain

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.After

/-- Reuse the derived SSA/heap adapter under the very same interpretation.
The only additional premise is a decidable syntax equation, not an equality
of executions, raw hashes, or unspecified operation meanings. -/
theorem actual_middle_complete {Value User : Type}
    (actual : Module) (swaps : List (Nat × Nat))
    (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (bound : plan.module = permuteModule swaps actual)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (scalar : Epoch.BinaryMeaning Value) (inputs : List (Runtime Value))
    (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers
      (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (Epoch.graphMeaning plan.opcodes scalar) actual false kernel.function.name inputs
      ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (Epoch.finalClock 0 kernel.commands) after := by
  have rename := complete_permuted swaps (Epoch.graphMeaning plan.opcodes scalar) actual
    kernel.function.name inputs ⟨⟨0, user⟩, heap⟩ outputs world
  rw [← bound] at rename
  exact rename.trans (kernel.complete_equivalent plan planValid facts scalar inputs typed callers heap user outputs world)

theorem actual_middle_observations {Value User : Type}
    (actual : Module) (swaps : List (Nat × Nat))
    (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (bound : plan.module = permuteModule swaps actual)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (scalar : Epoch.BinaryMeaning Value) (inputs : List (Runtime Value))
    (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers
      (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) actual
      kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar false false
        (Epoch.entryMemory plan kernel.function inputs heap user) observed := by
  have renamed : Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) actual
      kernel.function.name inputs heap user observed ↔
      Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) plan.module
        kernel.function.name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr
    intro returned
    apply exists_congr
    intro world
    apply and_congr_left
    intro _
    rw [bound]
    exact complete_permuted swaps _ actual _ inputs _ returned world
  exact renamed.trans (kernel.observations plan planValid facts scalar inputs typed callers heap user observed)

/-- Explicitly a typed actual-middle to epoch-blueprint result. This must not
be reported as ordinary evaluation of an independently decoded final SCF IR. -/
theorem actual_middle_epoch_chain {Value User : Type}
    (actual : Module) (swaps : List (Nat × Nat))
    (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (bound : plan.module = permuteModule swaps actual)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (memory : GraphLoopsMemorySCF.Model.Projection) (binding : Epoch.MemoryBinding plan kernel memory)
    (accepted : GraphLoopsMemorySCF.Model.check memory = true)
    (scalar : Epoch.BinaryMeaning Value) (inputs : List (Runtime Value))
    (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers
      (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) actual
      kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed := by
  exact (actual_middle_observations actual swaps plan planValid bound kernel facts scalar inputs typed callers
    heap user observed).trans (Epoch.epoch_chain plan kernel memory binding accepted scalar
      (Epoch.entryMemory plan kernel.function inputs heap user)
      ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ rfl observed)

/-- The independently checked shape pair supplies the complete first edge.
The final relation deliberately remains an epoch model outcome, not a claimed
execution of the raw reused/segmented IR. -/
theorem checked_source_epoch_chain {Value User : Type} (projection : Projection)
    (shapeAccepted : check projection = true)
    (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (bound : plan.module = permuteModule projection.swaps projection.after)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (memory : GraphLoopsMemorySCF.Model.Projection) (binding : Epoch.MemoryBinding plan kernel memory)
    (memoryAccepted : GraphLoopsMemorySCF.Model.check memory = true)
    (scalar : Epoch.BinaryMeaning Value) (inputs : List (Runtime Value))
    (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers
      (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) projection.before
      kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed := by
  have first : Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) projection.before
      kernel.function.name inputs heap user observed ↔
      Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) projection.after
        kernel.function.name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr
    intro returned
    apply exists_congr
    intro world
    apply and_congr_left
    intro _
    exact checked_complete shapeAccepted _ _ inputs _ returned world
  exact first.trans (actual_middle_epoch_chain projection.after projection.swaps plan planValid bound
    kernel facts memory binding memoryAccepted scalar inputs typed callers heap candidateHeap user observed)

end Merlin.GraphLoopsShape.After
