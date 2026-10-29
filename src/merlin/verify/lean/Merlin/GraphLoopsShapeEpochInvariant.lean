/- A checked statement preserves the scoped typed SSA and private/caller view
invariants needed by every later statement. In particular, initializers cannot
borrow an assumed runtime input relation independent of the preceding program. -/
import Merlin.GraphLoopsShapeEpochCommand

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem Command.assignment {Value World : Type} (plan : PlanContext) (context : Context) (clock : Nat)
    (command : Command) (checked : command.Valid plan context clock) (meaning : Interpretation Value World)
    (env nextEnv : Env Value) (before after : World) (run : command.atom.step meaning env before nextEnv after) :
    ∃ values, ValuesHold command.results values ∧ nextEnv = assign env command.results values := by
  cases command with
  | constant key result value =>
    exact ⟨[.index value], .cons (result.holds_index value checked.1.2.1 checked.1.2.2) .nil, run.1⟩
  | binary => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | allocate => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | release => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | «initialize» => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | initializeLine => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | load => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | store => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩
  | observer => obtain ⟨values, _, typed, assigned⟩ := run; exact ⟨values, typed, assigned⟩

theorem Command.pins_preserved {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (context : Context) (clock : Nat) (command : Command) (checked : command.Valid plan context clock)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User))
    (run : command.atom.step (plan.meaning scalar depth) before.observed.registers
      (clockWorld plan.objects.length clock before) env world) : Pinned plan.views env := by
  obtain ⟨values, typed, assigned⟩ := command.assignment plan context clock checked _ _ _ _ _ run
  cases command with
  | allocate operation item retained =>
    obtain ⟨after, oldRun, envEqual, _worldEqual⟩ :=
      (Command.correspondence plan planValid context clock (.allocate operation item retained) checked scalar depth
        before registers callers env world).mp run
    have final := (allocate_step_iff plan.objects plan.configs (plan.oldMeaning scalar) item.object retained before after).mp oldRun
    rw [final] at envEqual
    have same : env = before.observed.registers := by
      simpa only [GraphLoopsReuse.clearMemory, Bool.false_and, Bool.false_eq_true, ↓reduceIte] using envEqual
    rw [same]
    exact registers.2
  | constant | binary | release | «initialize» | initializeLine | load | store | observer =>
    rw [assigned]
    exact pinned_assign plan.views before.observed.registers registers.2 _ values checked.2.2.2.1

theorem Command.preserves {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (context : Context) (clock : Nat) (command : Command) (checked : command.Valid plan context clock)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User))
    (run : command.atom.step (plan.meaning scalar depth) before.observed.registers
      (clockWorld plan.objects.length clock before) env world) :
    RegistersValid (extendContext context command.results) plan.views env ∧
      CallersValid plan.objects.length plan.callers env := by
  obtain ⟨values, typed, assigned⟩ := command.assignment plan context clock checked _ _ _ _ _ run
  refine ⟨⟨?_, command.pins_preserved plan planValid context clock checked scalar depth before registers callers env world run⟩, ?_⟩
  · rw [assigned]
    exact typed_assign context before.observed.registers registers.1 command.results checked.2.2.1 values typed
  · rw [assigned]
    exact callers_assign plan.objects.length context plan.callers before.observed.registers callers checked.2.2.2.2
      command.results checked.2.2.1 values

end Merlin.GraphLoopsShape.Epoch
