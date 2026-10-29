import EpochKernel
namespace GraphShapeEpochObligation
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
open Merlin Merlin.GraphLoopsShape
theorem complete_adapter {Value User : Type} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (Epoch.graphMeaning plan.opcodes scalar) plan.module false kernel.function.name inputs
      ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (Epoch.finalClock 0 kernel.commands) after :=
    kernel.complete_equivalent plan (of_decide_eq_true plan_valid) (of_decide_eq_true kernel_facts)
      scalar inputs typed callers heap user outputs world
theorem memory_binding : decide (Epoch.MemoryBinding plan kernel GraphMemorySCFObligation.actual) = true := by decide
theorem complete_chain {Value User : Type} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar)
      GraphShapeObligation.actual.module kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed :=
    kernel.checked_chain GraphShapeObligation.actual.module GraphShapeObligation.actual.bits
      GraphShapeObligation.valid.1 plan (of_decide_eq_true plan_valid) normalized_syntax
      (of_decide_eq_true kernel_facts) GraphMemorySCFObligation.actual
      (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted scalar inputs typed callers
      heap candidateHeap user observed

end GraphShapeEpochObligation
