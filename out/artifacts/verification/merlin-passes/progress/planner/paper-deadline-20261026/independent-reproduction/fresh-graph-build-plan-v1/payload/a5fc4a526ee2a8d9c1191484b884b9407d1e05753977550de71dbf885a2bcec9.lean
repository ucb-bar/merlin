/- Whole ordered statement execution correspondence. The induction covers every
retained/changed statement and carries scoped registers, caller descriptors,
allocation naming and the complete modeled memory world across each boundary. -/
import Merlin.GraphLoopsShapeEpochInvariant

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def finalClock : Nat → List Command → Nat
  | clock, [] => clock
  | clock, command :: rest => finalClock (command.nextClock clock) rest

def finalContext : Context → List Command → Context
  | context, [] => context
  | context, command :: rest => finalContext (extendContext context command.results) rest

theorem commands_forward {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (commands : List Command) (context : Context) (clock : Nat) (valid : CommandsValid plan context clock commands)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User))
    (run : SequenceRun (plan.meaning scalar depth) (commands.map Command.atom) before.observed.registers
      (clockWorld plan.objects.length clock before) env world) :
    ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
      (commands.map Command.instruction) before after ∧ env = after.observed.registers ∧
      world = clockWorld plan.objects.length (finalClock clock commands) after ∧
      RegistersValid (finalContext context commands) plan.views after.observed.registers ∧
      CallersValid plan.objects.length plan.callers after.observed.registers := by
  induction commands generalizing context clock before with
  | nil =>
    cases run
    exact ⟨before, .done before, rfl, rfl, registers, callers⟩
  | cons command rest induction =>
    cases run with
    | more first suffix =>
      obtain ⟨middle, oldFirst, envEqual, worldEqual⟩ :=
        (command.correspondence plan planValid context clock valid.1 scalar depth before registers callers _ _).mp first
      have post := command.preserves plan planValid context clock valid.1 scalar depth before registers callers _ _ first
      rw [envEqual] at post
      rw [envEqual, worldEqual] at suffix
      obtain ⟨after, oldSuffix, finalEnv, finalWorld, finalRegisters, finalCallers⟩ :=
        induction (extendContext context command.results) (command.nextClock clock) valid.2 middle post.1 post.2 suffix
      exact ⟨after, .more oldFirst oldSuffix, finalEnv, finalWorld, finalRegisters, finalCallers⟩

theorem commands_backward {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (commands : List Command) (context : Context) (clock : Nat) (valid : CommandsValid plan context clock commands)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (run : GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
      (commands.map Command.instruction) before after) :
    SequenceRun (plan.meaning scalar depth) (commands.map Command.atom) before.observed.registers
      (clockWorld plan.objects.length clock before) after.observed.registers
      (clockWorld plan.objects.length (finalClock clock commands) after) ∧
      RegistersValid (finalContext context commands) plan.views after.observed.registers ∧
      CallersValid plan.objects.length plan.callers after.observed.registers := by
  induction commands generalizing context clock before with
  | nil =>
    cases run
    exact ⟨.done _ _, registers, callers⟩
  | cons command rest induction =>
    cases run with
    | more first suffix =>
      have sourceFirst := (command.correspondence plan planValid context clock valid.1 scalar depth
        before registers callers _ _).mpr ⟨_, first, rfl, rfl⟩
      have post := command.preserves plan planValid context clock valid.1 scalar depth before registers callers _ _ sourceFirst
      obtain ⟨sourceSuffix, finalRegisters, finalCallers⟩ :=
        induction (extendContext context command.results) (command.nextClock clock) valid.2 _ post.1 post.2 suffix
      exact ⟨.more sourceFirst sourceSuffix, finalRegisters, finalCallers⟩

/-- The whole ordered source block and old epoch instruction sequence have the
same complete executions under the derived state relation, in BOTH directions. -/
theorem commands_equivalent {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (commands : List Command) (context : Context) (clock : Nat) (valid : CommandsValid plan context clock commands)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    SequenceRun (plan.meaning scalar depth) (commands.map Command.atom) before.observed.registers
      (clockWorld plan.objects.length clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        (commands.map Command.instruction) before after ∧ env = after.observed.registers ∧
        world = clockWorld plan.objects.length (finalClock clock commands) after := by
  constructor
  · intro run
    obtain ⟨after, run, env, world, _, _⟩ :=
      commands_forward plan planValid commands context clock valid scalar depth before registers callers env world run
    exact ⟨after, run, env, world⟩
  · rintro ⟨after, run, rfl, rfl⟩
    exact (commands_backward plan planValid commands context clock valid scalar depth before after registers callers run).1

end Merlin.GraphLoopsShape.Epoch
