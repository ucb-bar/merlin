/- Derived operation and call meanings for a checked complete statement plan.
Finite key/configuration checks justify the local hypotheses used by the step
lemmas; no compiler-provided whole-program equivalence is admitted. -/
import Merlin.GraphLoopsShapeEpochPlan

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def PlanContext.Valid (plan : PlanContext) : Prop :=
  (∀ first ∈ plan.configs, ∀ second ∈ plan.configs, first.key = second.key → first = second) ∧
  ∀ entry ∈ plan.opcodes, entry.2 = .load ∨ entry.2 = .store → callTarget plan.module.calls entry.1 = none

instance planContextValidDecidable (plan : PlanContext) : Decidable plan.Valid :=
  inferInstanceAs (Decidable (_ ∧ _))

theorem opcodeAt_member (opcodes : Opcodes) (key : String) (opcode : Opcode)
    (found : opcodeAt opcodes key = some opcode) : (key, opcode) ∈ opcodes := by
  induction opcodes with
  | nil => cases found
  | cons entry rest induction =>
    simp only [opcodeAt] at found
    split at found
    · rename_i equal
      have actual := Option.some.inj found
      cases entry
      subst_vars
      exact List.mem_cons_self
    · exact List.mem_cons_of_mem entry (induction found)

theorem PlanContext.memory_nonCall (plan : PlanContext) (valid : plan.Valid) (key : String)
    (classified : memoryOpcode plan.opcodes key = .load ∨ memoryOpcode plan.opcodes key = .store) :
    callTarget plan.module.calls key = none := by
  cases found : opcodeAt plan.opcodes key with
  | none => simp only [memoryOpcode, found] at classified; cases classified <;> contradiction
  | some opcode =>
    cases opcode with
    | load => exact valid.2 (key, .load) (opcodeAt_member plan.opcodes key .load found) (Or.inl rfl)
    | store => exact valid.2 (key, .store) (opcodeAt_member plan.opcodes key .store found) (Or.inr rfl)
    | loop => simp only [memoryOpcode, found] at classified; cases classified <;> contradiction
    | binary element => simp only [memoryOpcode, found] at classified; cases classified <;> contradiction
    | allocate shape element => simp only [memoryOpcode, found] at classified; cases classified <;> contradiction
    | release => simp only [memoryOpcode, found] at classified; cases classified <;> contradiction

def PlanContext.oldMeaning {Value User : Type} (plan : PlanContext) (scalar : BinaryMeaning Value) :
    GraphLoopsMemorySCF.Interpretation Value (PublicState Value User) :=
  interpretation plan.objects.length plan.specifications plan.mapping scalar

/-- A kernel frame has at least one remaining observer-entry level. The actual
nonrecursive observer is proved independent of any larger finite call budget. -/
def PlanContext.meaning {Value User : Type} (plan : PlanContext) (scalar : BinaryMeaning Value) (depth : Nat) :
    Interpretation Value (MemoryWorld Value (AllocationPublic User)) :=
  withCalls (graphMeaning plan.opcodes scalar) plan.module
    (GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false (depth + 1))

theorem PlanContext.meaning_action {Value User : Type} (plan : PlanContext) (scalar : BinaryMeaning Value)
    (depth : Nat) (key : String) (notCall : callTarget plan.module.calls key = none) :
    (plan.meaning (User := User) scalar depth).action key = (graphMeaning plan.opcodes scalar).action key :=
  dispatched_action _ _ _ key notCall

theorem PlanContext.meaning_call {Value User : Type} (plan : PlanContext) (scalar : BinaryMeaning Value)
    (depth : Nat) (key name : String) (function : Function)
    (target : callTarget plan.module.calls key = some name) (found : functionAt plan.module name = some function) :
    (plan.meaning (User := User) scalar depth).action key =
      GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false (depth + 1) name := by
  simp only [PlanContext.meaning, withCalls, target, found, Option.isSome_some, ↓reduceIte]

theorem PlanContext.meaning_memory {Value User : Type} (plan : PlanContext) (valid : plan.Valid)
    (scalar : BinaryMeaning Value) (depth : Nat) :
    plan.meaning (User := User) scalar depth =
      withMemory (withCalls (scalarAllocationBase plan.opcodes scalar) plan.module
        (GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false (depth + 1))) (memoryOpcode plan.opcodes) :=
  calls_with_memory _ _ _ _ (plan.memory_nonCall valid)

end Merlin.GraphLoopsShape.Epoch
