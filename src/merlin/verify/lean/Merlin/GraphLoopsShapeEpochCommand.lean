/- Soundness of one checked actual statement correspondence. Every dynamic
obligation is derived from explicit operation semantics, scoped register/caller
invariants and recomputed finite plan facts. No whole-run premise is accepted. -/
import Merlin.GraphLoopsShapeEpochPlanMeaning

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem Command.correspondence {Value User : Type} (plan : PlanContext) (planValid : plan.Valid)
    (context : Context) (clock : Nat) (command : Command) (checked : command.Valid plan context clock)
    (scalar : BinaryMeaning Value) (depth : Nat)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (registers : RegistersValid context plan.views before.observed.registers)
    (callers : CallersValid plan.objects.length plan.callers before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    command.atom.step (plan.meaning scalar depth) before.observed.registers
      (clockWorld plan.objects.length clock before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) false false
        command.instruction before after ∧ env = after.observed.registers ∧
        world = clockWorld plan.objects.length (command.nextClock clock) after := by
  rcases checked with ⟨operations, bound, _fresh, _protects, _callersBound⟩
  cases command with
  | constant key result value =>
    exact constant_atom plan.objects.length clock plan.specifications scalar plan.mapping key result value
      operations.1 plan.objects plan.configs (plan.meaning scalar depth) before env world
  | binary key operation left right result element =>
    rcases operations with ⟨selected, opcode, notCall, _, _, _⟩
    have action : (plan.meaning (User := User) scalar depth).action operation = Binary element (scalar operation) :=
      (plan.meaning_action scalar depth operation notCall).trans (graph_binary plan.opcodes scalar operation element opcode)
    exact binary_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation left right result
      element selected plan.objects plan.configs (plan.meaning scalar depth) action before env world
  | allocate operation item retained =>
    rcases operations with ⟨itemValid, sameClock, opcode, notCall⟩
    subst clock
    have action : (plan.meaning (User := User) scalar depth).action operation = Allocate item.shape item.element :=
      (plan.meaning_action scalar depth operation notCall).trans
        (graph_allocate plan.opcodes scalar operation item.shape item.element opcode)
    exact allocate_atom plan.objects.length item.object itemValid.2.1 plan.objects plan.configs (plan.oldMeaning scalar)
      retained operation item.binding item.shape item.element itemValid.2.2.2.1 itemValid.2.2.2.2
      (plan.meaning scalar depth) action before (registers.2 item itemValid.1) env world
  | release operation item retained =>
    rcases operations with ⟨itemValid, opcode, notCall⟩
    have action : (plan.meaning (User := User) scalar depth).action operation = Release :=
      (plan.meaning_action scalar depth operation notCall).trans (graph_release plan.opcodes scalar operation opcode)
    exact release_atom plan.objects.length clock item.object itemValid.2.1 plan.objects plan.configs (plan.oldMeaning scalar)
      retained operation item.binding item.shape item.element (plan.meaning scalar depth) action before
      (registers.2 item itemValid.1) env world
  | «initialize» link =>
    rcases operations with ⟨facts, inputBound, configMember, mapping, inside, store, loop⟩
    have classifiedStore : memoryOpcode plan.opcodes link.canonical.storeKey = .store := by
      simp only [memoryOpcode, store]
    have classifiedLoop : memoryOpcode plan.opcodes link.canonical.loopKey = .loop := by
      simp only [memoryOpcode, loop]
    have unique : ∀ actual ∈ plan.configs, actual.key = link.config.key → actual = link.config :=
      fun actual member equal => planValid.1 actual member link.config configMember equal
    have step := initializer_atom link (decide_eq_true facts) context plan.views (decide_eq_true inputBound)
      plan.objects.length clock inside plan.objects plan.configs configMember unique plan.specifications scalar
      (withCalls (scalarAllocationBase plan.opcodes scalar) plan.module
        (GraphLoopsShape.invoke (graphMeaning plan.opcodes scalar) plan.module false (depth + 1)))
      (memoryOpcode plan.opcodes) classifiedStore classifiedLoop before registers env world
    rw [← plan.meaning_memory planValid scalar depth] at step
    simpa only [mapping, Command.atom, Command.instruction, Command.nextClock, PlanContext.oldMeaning] using step
  | load key operation access result =>
    rcases operations with ⟨selected, accessValid, opcode, notCall, _⟩
    obtain ⟨coordinates, indices⟩ := indices_of_typed context before.observed.registers registers.1 access.indices
      (fun binding member => bound binding (List.mem_cons_of_mem _ member)) accessValid.2
    have action : (plan.meaning (User := User) scalar depth).action operation = Load :=
      (plan.meaning_action scalar depth operation notCall).trans (graph_load plan.opcodes scalar operation opcode)
    exact load_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation access result selected
      plan.objects plan.configs accessValid.1.2.2.1 accessValid.1.2.1 (plan.meaning scalar depth) action before
      (registers.2 _ accessValid.1.1) coordinates indices env world
  | store key operation access operand =>
    rcases operations with ⟨selected, accessValid, opcode, notCall, operandScalar⟩
    obtain ⟨coordinates, indices⟩ := indices_of_typed context before.observed.registers registers.1 access.indices
      (fun binding member => bound binding (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))) accessValid.2
    obtain ⟨value, scalarRead⟩ := scalar_of_kind (before.observed.registers operand.name) access.element
      ((registers.1 operand (bound operand (List.mem_cons_self))).1.trans operandScalar.1)
    have action : (plan.meaning (User := User) scalar depth).action operation = Store :=
      (plan.meaning_action scalar depth operation notCall).trans (graph_store plan.opcodes scalar operation opcode)
    exact store_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation access operand selected
      plan.objects plan.configs accessValid.1.2.2.1 accessValid.1.2.1 (plan.meaning scalar depth) action before
      (registers.2 _ accessValid.1.1) coordinates indices value scalarRead env world
  | observer key operation name input buffer result program =>
    rcases operations with ⟨selected, programFacts, found, separate, target, loadNotCall, binaryNotCall, storeNotCall,
      loadOpcode, storeOpcode, binaryOpcode, inputScalar, _resultScalar, callerMember⟩
    obtain ⟨value, inputRead⟩ := scalar_of_kind (before.observed.registers input.name) program.element
      ((registers.1 input (bound input (List.mem_cons_self))).1.trans inputScalar.1)
    obtain ⟨object, outside, bufferRead⟩ := callers _ callerMember
    exact observer_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation name input buffer result
      program (decide_eq_true programFacts) selected plan.module found separate loadNotCall binaryNotCall storeNotCall
      (graphMeaning plan.opcodes scalar) (graph_load plan.opcodes scalar program.loadKey loadOpcode)
      (graph_binary plan.opcodes scalar program.binaryKey program.element binaryOpcode)
      (graph_store plan.opcodes scalar program.storeKey storeOpcode) (plan.meaning scalar depth) depth
      (plan.meaning_call scalar depth operation name _ target found) plan.objects plan.configs before object outside value
      inputRead bufferRead env world

end Merlin.GraphLoopsShape.Epoch
