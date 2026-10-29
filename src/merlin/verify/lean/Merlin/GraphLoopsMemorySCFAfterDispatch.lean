/- UNCOMPILED. Actual expanded-module call dispatch and generated-index
selection. Observer bodies are selected by finite functionAt equalities in the
expanded module, not by reusing the source module's recursive evaluator.
-/
import Merlin.GraphLoopsMemorySCFAfterBoundary
import Merlin.GraphLoopsMemorySCFAfterScopedInitializer

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def expandedMeaning {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)) :=
  withCalls (finalMeaning bits indices plan.opcodes scalar) (expandedModule plan kernel expansions)
    (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
      (expandedModule plan kernel expansions) false (depth + 1))

def ObserverBinding (target : Module) : Epoch.Command → Prop
  | .observer _ _ name _ _ _ program => functionAt target name = some (program.function name)
  | _ => True

instance (target : Module) (command : Epoch.Command) : Decidable (ObserverBinding target command) := by
  cases command <;> unfold ObserverBinding <;> infer_instance

/-- All these are recomputed finite facts. In particular, candidate physical
slots and every actual observer body are checked rather than assumed at runtime.
-/
def BoundaryFacts (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) : Prop :=
  (∀ object ∈ List.range plan.objects.length, GraphLoopsReuse.slotOf plan.objects object < plan.objects.length) ∧
    EntrySlotFacts plan kernel.function ∧
    AllBound (Epoch.finalContext kernel.function.parameters kernel.commands) kernel.returned ∧
    (∀ command ∈ kernel.commands, ObserverBinding (expandedModule plan kernel expansions) command) ∧
    functionAt (expandedModule plan kernel expansions) kernel.function.name =
      some { kernel.function with code := fragmentsCode (expansions.flatMap Expansion.fragments) kernel.returned }

instance (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) :
    Decidable (BoundaryFacts plan kernel expansions) := by
  unfold BoundaryFacts
  infer_instance

theorem expanded_primitive {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (noOverrides : NoOpcodeOverrides plan indices) (scalar : Epoch.BinaryMeaning Value) (depth : Nat)
    (key : String) (opcode : Epoch.Opcode) (selected : Epoch.opcodeAt plan.opcodes key = some opcode)
    (notCall : callTarget plan.module.calls key = none) :
    (expandedMeaning (User := User) bits indices plan kernel expansions scalar depth).action key =
      (Epoch.graphMeaning plan.opcodes scalar).action key := by
  have unchanged := noOverrides.1 (key, opcode) (Epoch.opcodeAt_member plan.opcodes key opcode selected)
  exact (Epoch.dispatched_action _ (expandedModule plan kernel expansions) _ key notCall).trans
    (unchanged_action bits indices plan.opcodes scalar key unchanged)

theorem expanded_generated {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) (key : String) (opcode : IndexOpcode)
    (selected : indexOpcodeAt indices key = some opcode)
    (notOld : Epoch.opcodeAt plan.opcodes key = none) (notCall : callTarget plan.module.calls key = none) :
    (expandedMeaning (User := User) bits indices plan kernel expansions scalar depth).action key =
      IndexAction bits opcode := by
  exact (Epoch.dispatched_action _ (expandedModule plan kernel expansions) _ key notCall).trans
    (generated_action bits indices plan.opcodes scalar key opcode selected
      (by simp only [Epoch.memoryOpcode, notOld]))

theorem expanded_call {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) (key name : String) (function : Function)
    (target : callTarget plan.module.calls key = some name)
    (found : functionAt (expandedModule plan kernel expansions) name = some function) :
    (expandedMeaning (User := User) bits indices plan kernel expansions scalar depth).action key =
      GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
        (expandedModule plan kernel expansions) false (depth + 1) name := by
  simp only [expandedMeaning, withCalls, expanded_calls, target, found, Option.isSome_some, ↓reduceIte]

theorem expanded_memory {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (valid : plan.Valid) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    expandedMeaning (User := User) bits indices plan kernel expansions scalar depth =
      withMemory (withCalls (withIndex (Epoch.scalarAllocationBase plan.opcodes scalar) bits indices)
        (expandedModule plan kernel expansions)
        (GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar)
          (expandedModule plan kernel expansions) false (depth + 1))) (Epoch.memoryOpcode plan.opcodes) :=
  Epoch.calls_with_memory _ _ _ _ (plan.memory_nonCall valid)

end Merlin.GraphLoopsMemorySCF.After
