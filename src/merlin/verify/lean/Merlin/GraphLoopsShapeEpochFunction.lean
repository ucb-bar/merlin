/- From actual function entry to the complete epoch program. All finite call
depths are covered: an unconditional internal call refuses depth one, while
the concrete observer proof is independent of every positive callee depth. -/
import Merlin.GraphLoopsShapeEpochReturn
import Merlin.GraphLoopsShapeEpochEntry

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure Kernel where
  function : Function
  commands : List Command
  returnKey : String
  returned : List Binding
  deriving DecidableEq, Repr

def Command.CallEntry (plan : PlanContext) : Command → Prop
  | .observer _ operation name _ _ _ _ =>
      callTarget plan.module.calls operation = some name ∧ (functionAt plan.module name).isSome = true
  | _ => False

instance commandCallEntryDecidable (plan : PlanContext) (command : Command) : Decidable (command.CallEntry plan) := by
  cases command <;> unfold Command.CallEntry <;> infer_instance

def Kernel.Facts (plan : PlanContext) (kernel : Kernel) : Prop :=
  functionAt plan.module kernel.function.name = some kernel.function ∧
  kernel.function.code = sequenceCode (kernel.commands.map Command.atom) kernel.returned ∧
  CommandsValid plan kernel.function.parameters 0 kernel.commands ∧
  specAt plan.specifications kernel.returnKey = some (.returned kernel.returned) ∧
  EntryFacts plan kernel.function ∧ ∃ command ∈ kernel.commands, command.CallEntry plan

instance kernelFactsDecidable (plan : PlanContext) (kernel : Kernel) : Decidable (kernel.Facts plan) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def Kernel.instructions (kernel : Kernel) : List GraphLoopsReuse.Instruction :=
  kernel.commands.map Command.instruction ++ [.retained kernel.returnKey]

theorem zero_call_refused {Value World : Type} (base : Interpretation Value World) (module : Module)
    (key name : String) (target : callTarget module.calls key = some name)
    (found : (functionAt module name).isSome = true) (operands results : List Binding)
    (env next : Env Value) (before after : World) :
    ¬ (Atom.action key operands results).step (withCalls base module (GraphLoopsShape.invoke base module false 0))
      env before next after := by
  rintro ⟨values, action, _, _⟩
  simp only [withCalls, target, found, ↓reduceIte, GraphLoopsShape.invoke] at action

theorem Kernel.depth_one_refused {Value User : Type} (plan : PlanContext) (kernel : Kernel)
    (facts : kernel.Facts plan) (scalar : BinaryMeaning Value) (inputs : List (Runtime Value))
    (before after : MemoryWorld Value (AllocationPublic User)) (outputs : List (Runtime Value)) :
    ¬ GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false 1 kernel.function.name
      inputs before outputs after := by
  simp only [GraphLoopsShape.invoke, facts.1]
  rintro ⟨_, run⟩
  have plain := (eval_aliases _ kernel.function.code _ [] _ _ _ _).mp run
  rw [facts.2.1] at plain
  obtain ⟨env, sequence, _⟩ := (sequence_complete _ _ _ _ _ _ _).mp plain
  obtain ⟨command, member, call⟩ := facts.2.2.2.2.2
  have refuses : ∀ (initial : Env Value) (world : MemoryWorld Value (AllocationPublic User))
      (final : Env Value) (after : MemoryWorld Value (AllocationPublic User)), ¬ command.atom.step
      (withCalls (graphMeaning plan.opcodes scalar) plan.module
        (GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false 0)) initial world final after := by
    cases command with
    | observer key operation name input buffer result program =>
      intro initial world final after
      exact zero_call_refused _ _ operation name call.1 call.2 [input, buffer] [result] initial final world after
    | constant | binary | allocate | release | «initialize» | load | store => exact False.elim call
  exact sequence_refused _ _ command.atom (List.mem_map_of_mem member) refuses _ env before after sequence

theorem Kernel.depth_equivalent {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (kernel : Kernel) (facts : kernel.Facts plan) (scalar : BinaryMeaning Value) (depth : Nat)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (AllocationPublic User)) :
    GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false (depth + 2)
      kernel.function.name inputs ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = clockWorld plan.objects.length (finalClock 0 kernel.commands) after := by
  simp only [GraphLoopsShape.invoke, facts.1, Function.invoke, typed, true_and]
  change Eval (plan.meaning scalar depth) false (extendAliases [] kernel.function.parameters) kernel.function.code
    (assign emptyEnv kernel.function.parameters inputs) ⟨⟨0, user⟩, heap⟩ outputs world ↔ _
  rw [eval_aliases _ kernel.function.code _ []]
  rw [seed_eval _ kernel.function.code _ plan.views facts.2.2.2.2.1.2.2.2.1]
  rw [facts.2.1]
  have whole := block_equivalent plan planValid kernel.commands kernel.function.parameters 0 facts.2.2.1
    kernel.returnKey kernel.returned facts.2.2.2.1 scalar depth
    (entryMemory plan kernel.function inputs heap user)
    (entry_registers plan kernel.function facts.2.2.2.2.1 inputs typed heap user)
    (entry_callers plan kernel.function facts.2.2.2.2.1 inputs callers heap user) outputs world
  rw [entry_world] at whole
  exact whole

/-- Quantification over finite derivations is exact; the theorem does not fix
one arbitrary call fuel or assume success of partial scalar/caller actions. -/
theorem Kernel.complete_equivalent {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (kernel : Kernel) (facts : kernel.Facts plan) (scalar : BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (AllocationPublic User)) :
    complete (graphMeaning plan.opcodes scalar) plan.module false kernel.function.name inputs
      ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = clockWorld plan.objects.length (finalClock 0 kernel.commands) after := by
  constructor
  · rintro ⟨depth, run⟩
    cases depth with
    | zero => exact False.elim run
    | succ depth =>
      cases depth with
      | zero => exact False.elim (kernel.depth_one_refused plan facts scalar inputs _ _ outputs run)
      | succ depth => exact (kernel.depth_equivalent plan planValid facts scalar depth inputs typed callers heap user outputs world).mp run
  · intro run
    exact ⟨2, (kernel.depth_equivalent plan planValid facts scalar 0 inputs typed callers heap user outputs world).mpr run⟩

end Merlin.GraphLoopsShape.Epoch
