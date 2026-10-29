/- UNCOMPILED. Derived localLaw simulations for retained constants, partial
arithmetic and actual internal Touch calls. Arithmetic is never reassociated or
made total; the observer proof executes the expanded module's selected body.
-/
import Merlin.GraphLoopsMemorySCFAfterDispatch

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem fragments_single {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) (fragment : Fragment)
    (actual next : GraphLoopsShape.Env Value) (before after : World) :
    FragmentsRun meaning [fragment] actual before next after ↔ fragment.step meaning actual before next after := by
  constructor
  · intro run; cases run with | more first rest => cases rest; exact first
  · intro run; exact .more run (.done _ _)

/-- Generic single-action lifting; the only semantic premise is a LOCAL old
action formula. The binary/observer theorems below derive it from frozen roots.
-/
theorem retained_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (command : Epoch.Command)
    (checked : command.Valid plan context sourceClock) (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (key operation : String) (operands results : List Binding) (spec : Epoch.Spec)
    (atomEqual : command.atom = .action operation operands results)
    (instructionEqual : command.instruction = .retained key) (resultsEqual : command.results = results)
    (clockEqual : (Expansion.keep command).nextClock clock = clock)
    (selected : Epoch.specAt plan.specifications key = some spec) (noViews : NoViewBindings results)
    (arguments : ∀ (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) (actual : GraphLoopsShape.Env Value),
      Epoch.RegistersValid context plan.views before.observed.registers →
      Epoch.CallersValid plan.objects.length plan.callers before.observed.registers →
      RegistersRelated plan.objects context before.observed.registers actual →
      readArguments false [] actual operands = readArguments false [] before.observed.registers operands)
    (localLaw : ∀ (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)),
      Epoch.RegistersValid context plan.views before.observed.registers →
      Epoch.CallersValid plan.objects.length plan.callers before.observed.registers →
      ∀ next world, (Epoch.Atom.action operation operands results).step meaning before.observed.registers
        (Epoch.clockWorld plan.objects.length clock before) next world ↔
        ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) false false
          (.retained key) before after ∧ next = after.observed.registers ∧
          world = Epoch.clockWorld plan.objects.length clock after) :
    LocalSimulation (User := User) plan scalar meaning context clock (.keep command) := by
  have fresh : Fresh context results := resultsEqual ▸ checked.2.2.1
  constructor
  · intro before after actual registers callers related run
    rw [Expansion.source, instructionEqual] at run
    obtain ⟨next, actualRun, finalRelated⟩ := retained_scoped_forward plan context clock scalar meaning key operation spec
      selected operands results fresh noViews before after actual related
      (arguments before actual registers callers related) (localLaw before registers callers) run
    refine ⟨next, ?_, ?_⟩
    · rw [Expansion.fragments, atomEqual, clockEqual, fragments_single]
      exact actualRun
    · simpa only [Expansion.source, resultsEqual] using finalRelated
  · intro before actual next world registers callers related run
    rw [Expansion.fragments, atomEqual, fragments_single] at run
    obtain ⟨after, oldRun, worldEqual, finalRelated⟩ := retained_scoped_backward plan context clock scalar meaning key operation
      spec selected operands results fresh noViews before actual next related
      (arguments before actual registers callers related) world (localLaw before registers callers) run
    refine ⟨after, ?_, ?_, ?_⟩
    · simpa only [Expansion.source, instructionEqual] using oldRun
    · simpa only [clockEqual] using worldEqual
    · simpa only [Expansion.source, resultsEqual] using finalRelated

theorem constant_local {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock clock : Nat) (key : String) (result : Binding) (value : Nat)
    (checked : (Epoch.Command.constant key result value).Valid plan context sourceClock)
    (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) :
    LocalSimulation (User := User) plan scalar meaning context clock (.keep (.constant key result value)) := by
  have normalized := candidate_retained_iff (User := User) plan.objects.length plan.specifications scalar plan.mapping
    key (.constant result value) checked.1.1 plan.objects plan.configs
  constructor
  · intro before after actual registers callers related run
    obtain ⟨observed, equal, rfl⟩ := (normalized before after).mp run
    change observed = Epoch.writeRegister before.observed result.name (.index value) at equal
    subst observed
    exact ⟨GraphLoopsShape.put actual result.name (.index value), .more ⟨rfl, rfl⟩ (.done _ _),
      related.define result (.index value) checked.2.2.1.1⟩
  · intro before actual next world registers callers related run
    have first := (fragments_single meaning (.atom (.constant result value)) actual next _ world).mp run
    obtain ⟨rfl, rfl⟩ := first
    refine ⟨⟨Epoch.writeRegister before.observed result.name (.index value), before.heap⟩,
      (normalized before _).mpr ⟨_, rfl, rfl⟩, rfl, ?_⟩
    exact related.define result (.index value) checked.2.2.1.1

theorem binary_local {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (noOverrides : NoOpcodeOverrides plan indices)
    (context : Context) (sourceClock clock : Nat) (key operation : String) (left right result : Binding) (element : String)
    (checked : (Epoch.Command.binary key operation left right result element).Valid plan context sourceClock)
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    LocalSimulation (User := User) plan scalar (expandedMeaning bits indices plan kernel expansions scalar depth)
      context clock (.keep (.binary key operation left right result element)) := by
  rcases checked.1 with ⟨selected, opcode, notCall, leftScalar, rightScalar, resultScalar⟩
  have action := (expanded_primitive (User := User) bits indices plan kernel expansions noOverrides scalar depth
    operation (.binary element) opcode notCall).trans (Epoch.graph_binary plan.opcodes scalar operation element opcode)
  apply retained_local plan context sourceClock clock (.binary key operation left right result element) checked scalar
    (expandedMeaning bits indices plan kernel expansions scalar depth) key operation [left, right] [result]
    (.binary operation left right result element) rfl rfl rfl rfl selected
  · intro binding member
    have same : binding = result := List.mem_singleton.mp member
    subst binding
    rw [resultScalar.1]
    trivial
  · intro before actual registers callers related
    apply related.returns registers.1 [left, right] checked.2.1
    intro binding member
    rcases List.mem_cons.mp member with rfl | rightMember
    · rw [leftScalar.1]; trivial
    · have same : binding = right := List.mem_singleton.mp rightMember
      subst binding
      rw [rightScalar.1]; trivial
  · intro before registers callers next world
    exact Epoch.binary_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation
      left right result element selected plan.objects plan.configs _ action before next world

theorem observer_local {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion)
    (noOverrides : NoOpcodeOverrides plan indices)
    (context : Context) (sourceClock clock : Nat) (key operation name : String)
    (input buffer result : Binding) (program : Epoch.Touch)
    (checked : (Epoch.Command.observer key operation name input buffer result program).Valid plan context sourceClock)
    (found : functionAt (expandedModule plan kernel expansions) name = some (program.function name))
    (scalar : Epoch.BinaryMeaning Value) (depth : Nat) :
    LocalSimulation (User := User) plan scalar (expandedMeaning bits indices plan kernel expansions scalar depth)
      context clock (.keep (.observer key operation name input buffer result program)) := by
  rcases checked.1 with ⟨selected, facts, _, separate, target, loadNotCall, binaryNotCall, storeNotCall,
    loadOpcode, storeOpcode, binaryOpcode, inputScalar, resultScalar, callerMember⟩
  have load := (unchanged_action (User := User) bits indices plan.opcodes scalar program.loadKey
    (noOverrides.1 _ (Epoch.opcodeAt_member _ _ _ loadOpcode))).trans
      (Epoch.graph_load plan.opcodes scalar program.loadKey loadOpcode)
  have binary := (unchanged_action (User := User) bits indices plan.opcodes scalar program.binaryKey
    (noOverrides.1 _ (Epoch.opcodeAt_member _ _ _ binaryOpcode))).trans
      (Epoch.graph_binary plan.opcodes scalar program.binaryKey program.element binaryOpcode)
  have store := (unchanged_action (User := User) bits indices plan.opcodes scalar program.storeKey
    (noOverrides.1 _ (Epoch.opcodeAt_member _ _ _ storeOpcode))).trans
      (Epoch.graph_store plan.opcodes scalar program.storeKey storeOpcode)
  apply retained_local plan context sourceClock clock (.observer key operation name input buffer result program) checked scalar
    (expandedMeaning bits indices plan kernel expansions scalar depth) key operation [input, buffer] [result]
    (.observer program.binaryKey input buffer result program.element) rfl rfl rfl rfl selected
  · intro binding member
    have same : binding = result := List.mem_singleton.mp member
    subst binding
    rw [resultScalar.1]; trivial
  · intro before actual registers callers related
    have scalarRead := related.scalar input (checked.2.1 input (by simp [Epoch.Command.inputs]))
      (registers.1 input (checked.2.1 input (by simp [Epoch.Command.inputs])))
      (by rw [inputScalar.1]; trivial)
    obtain ⟨object, outside, bufferRead⟩ := callers _ callerMember
    have viewRead := related.caller_view buffer (checked.2.1 buffer (by simp [Epoch.Command.inputs]))
      object outside [1] program.element bufferRead
    change [actual input.name, actual buffer.name] = _
    rw [scalarRead, viewRead]
    rfl
  · intro before registers callers next world
    obtain ⟨value, inputRead⟩ := Epoch.scalar_of_kind (before.observed.registers input.name) program.element
      ((registers.1 input (checked.2.1 input (by simp [Epoch.Command.inputs]))).1.trans inputScalar.1)
    obtain ⟨object, outside, bufferRead⟩ := callers _ callerMember
    exact Epoch.observer_atom plan.objects.length clock plan.specifications scalar plan.mapping key operation name
      input buffer result program (decide_eq_true facts) selected (expandedModule plan kernel expansions) found separate
      loadNotCall binaryNotCall storeNotCall (finalMeaning bits indices plan.opcodes scalar) load binary store
      (expandedMeaning bits indices plan kernel expansions scalar depth) depth
      (expanded_call bits indices plan kernel expansions scalar depth operation name _ target found)
      plan.objects plan.configs before object outside value inputRead bufferRead next world

end Merlin.GraphLoopsMemorySCF.After
