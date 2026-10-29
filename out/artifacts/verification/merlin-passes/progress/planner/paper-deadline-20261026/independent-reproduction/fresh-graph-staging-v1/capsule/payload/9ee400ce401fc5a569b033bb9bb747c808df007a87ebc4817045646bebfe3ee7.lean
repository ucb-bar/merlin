/- Complete normalized block correspondence, including its actual return.
The return observes the final scoped SSA values; it does not change memory.
This derives the whole boundary from the checked command simulation. -/
import Merlin.GraphLoopsShapeEpochExecution

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem execution_append {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List GraphLoopsMemorySCF.Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (candidate segmented : Bool) (initial suffix : List GraphLoopsReuse.Instruction)
    (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Execution objects configs meaning candidate segmented (initial ++ suffix) before after ↔
      ∃ middle, GraphLoopsMemorySCF.Execution objects configs meaning candidate segmented initial before middle ∧
        GraphLoopsMemorySCF.Execution objects configs meaning candidate segmented suffix middle after := by
  induction initial generalizing before with
  | nil =>
    constructor
    · intro run; exact ⟨before, .done before, run⟩
    · rintro ⟨middle, first, rest⟩; cases first; exact rest
  | cons instruction rest induction =>
    constructor
    · intro run
      cases run with
      | more first remaining =>
        obtain ⟨middle, initialRun, suffixRun⟩ := (induction _).mp remaining
        exact ⟨middle, .more first initialRun, suffixRun⟩
    · rintro ⟨middle, initialRun, suffixRun⟩
      cases initialRun with
      | more first remaining => exact .more first ((induction _).mpr ⟨middle, remaining, suffixRun⟩)

theorem execution_single {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List GraphLoopsMemorySCF.Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (candidate segmented : Bool) (instruction : GraphLoopsReuse.Instruction)
    (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Execution objects configs meaning candidate segmented [instruction] before after ↔
      GraphLoopsMemorySCF.Step objects configs meaning candidate segmented instruction before after := by
  constructor
  · intro run; cases run with | more first rest => cases rest; exact first
  · intro run; exact .more run (.done after)

def returnMemory {Value User : Type} (returned : List Binding)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsReuse.Memory Value (PublicState Value User) :=
  ⟨{ before.observed with returned := readArguments false [] before.observed.registers returned }, before.heap⟩

theorem return_world {Value User : Type} (bound clock : Nat) (returned : List Binding)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    clockWorld bound clock (returnMemory returned before) = clockWorld bound clock before := rfl

/-- Both directions preserve exact returned values and the entire modeled world.
No complete-execution equivalence is supplied as a premise. -/
theorem block_equivalent {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (commands : List Command) (context : Context) (clock : Nat) (valid : CommandsValid plan context clock commands)
    (returnKey : String) (returned : List Binding)
    (returnSelected : specAt plan.specifications returnKey = some (.returned returned))
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (AllocationPublic User)) :
    Eval (plan.meaning scalar depth) false [] (sequenceCode (commands.map Command.atom) returned)
      before.observed.registers (clockWorld plan.objects.length clock before) outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        (commands.map Command.instruction ++ [.retained returnKey]) before after ∧
        outputs = after.observed.returned ∧
        world = clockWorld plan.objects.length (finalClock clock commands) after := by
  rw [sequence_complete]
  constructor
  · rintro ⟨env, run, outputsEqual⟩
    obtain ⟨middle, oldRun, envEqual, worldEqual⟩ :=
      (commands_equivalent plan planValid commands context clock valid scalar depth before registers callers env world).mp run
    refine ⟨returnMemory returned middle, ?_, ?_, worldEqual⟩
    · apply (execution_append _ _ _ _ _ _ _ _ _).mpr
      refine ⟨middle, oldRun, (execution_single _ _ _ _ _ _ _ _).mpr ?_⟩
      exact (returned_step_iff plan.objects.length plan.specifications scalar plan.mapping returnKey returned
        returnSelected plan.objects plan.configs middle _).mpr rfl
    · simpa only [envEqual, returnMemory] using outputsEqual
  · rintro ⟨after, run, outputsEqual, worldEqual⟩
    obtain ⟨middle, initialRun, returnRun⟩ := (execution_append _ _ _ _ _ _ _ _ _).mp run
    have final := (returned_step_iff plan.objects.length plan.specifications scalar plan.mapping returnKey returned
      returnSelected plan.objects plan.configs middle after).mp ((execution_single _ _ _ _ _ _ _ _).mp returnRun)
    subst after
    refine ⟨middle.observed.registers, ?_, outputsEqual⟩
    apply (commands_equivalent plan planValid commands context clock valid scalar depth before registers callers _ world).mpr
    exact ⟨middle, initialRun, rfl, worldEqual⟩

end Merlin.GraphLoopsShape.Epoch
