/- UNCOMPILED. Before→actual-final graph observation candidate under ONE
operation interpretation, joining the frozen shape/reuse/segmentation roots to
the independently materialized final program. No numerical laws are introduced.
Kernel checking, actual finite obligations, review and runtime fidelity remain
outstanding; compilation of this file has NOT been observed.
-/
import Merlin.GraphLoopsMemorySCFAfterSupport
import Merlin.GraphLoopsShapeEpochChain

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem final_observations {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions)
    (actualFinal : Module) (swaps : List (Nat × Nat)) (sourceFrame finalFrame : List String)
    (materialized : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (finalMeaning bits indices plan.opcodes scalar) actualFinal kernel.function.name
      inputs heap user observed ↔ Epoch.epochOutcomes plan kernel scalar true true
        (Epoch.entryMemory plan kernel.function inputs heap user) observed := by
  constructor
  · rintro ⟨returned, world, run, observation⟩
    obtain ⟨after, oldRun, returnedEqual, worldEqual⟩ :=
      (actual_final_equivalent bits indices plan planValid kernel facts expansions binding boundary actualFinal swaps
        sourceFrame finalFrame materialized scalar inputs typed callers heap user returned world).mp run
    refine ⟨after.observed, ⟨after, oldRun, rfl⟩, ?_⟩
    rw [returnedEqual, worldEqual, Epoch.observe_clock] at observation
    exact observation
  · rintro ⟨state, ⟨after, run, stateEqual⟩, observation⟩
    refine ⟨after.observed.returned, Epoch.clockWorld plan.objects.length (physicalFinalClock 0 expansions) after, ?_, ?_⟩
    · exact (actual_final_equivalent bits indices plan planValid kernel facts expansions binding boundary actualFinal swaps
        sourceFrame finalFrame materialized scalar inputs typed callers heap user _ _).mpr ⟨after, run, rfl, rfl⟩
    · rw [Epoch.observe_clock, stateEqual]; exact observation

/-- The whole graph result derives from checked syntax/bindings and previous
per-operation roots, not a supplied source/target outcome equality. It compares
caller-visible possible successes; private allocation clocks may differ.
-/
theorem checked_actual_chain {Value User : Type} (source : Module) (bits : Nat) (sourceValid : source.Valid bits)
    (indices : IndexOpcodes) (sourceAbsent : SourceIndexAbsent indices source)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (normalized : plan.module = Epoch.normalizedModule source)
    (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (memory : GraphLoopsMemorySCF.Model.Projection) (memoryBinding : Epoch.MemoryBinding plan kernel memory)
    (memoryAccepted : GraphLoopsMemorySCF.Model.check memory = true)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions)
    (actualFinal : Module) (swaps : List (Nat × Nat)) (sourceFrame finalFrame : List String)
    (materialized : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (finalMeaning bits indices plan.opcodes scalar) source kernel.function.name
      inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (finalMeaning bits indices plan.opcodes scalar) actualFinal kernel.function.name
      inputs heap user observed := by
  have extension : Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar) source
      kernel.function.name inputs heap user observed ↔
      Epoch.graphOutcomes plan.objects.length (finalMeaning bits indices plan.opcodes scalar) source
        kernel.function.name inputs heap user observed := by
    unfold Epoch.graphOutcomes
    apply exists_congr; intro returned
    apply exists_congr; intro world
    apply and_congr_left; intro _
    exact source_complete_extension bits indices plan.opcodes scalar source sourceAbsent kernel.function.name
      inputs _ returned world
  have oldChain := kernel.checked_chain source bits sourceValid plan planValid normalized facts memory memoryBinding
    memoryAccepted scalar inputs typed callers heap heap user observed
  have actual := final_observations bits indices plan planValid kernel facts expansions binding boundary actualFinal swaps
    sourceFrame finalFrame materialized scalar inputs typed callers heap user observed
  exact extension.symm.trans (oldChain.trans actual.symm)

end Merlin.GraphLoopsMemorySCF.After
