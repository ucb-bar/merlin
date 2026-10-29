/- Finite, recomputed statement correspondence plan for the actual normalized
function and old epoch instructions. This is a checker input, not a semantic
certificate by assertion: subsequent soundness derives every dynamic step. -/
import Merlin.GraphLoopsShapeEpochAllocationSteps
import Merlin.GraphLoopsShapeEpochMemorySteps
import Merlin.GraphLoopsShapeEpochInitializerSteps
import Merlin.GraphLoopsShapeEpochObserverSteps

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure PlanContext where
  objects : GraphLoopsReuse.Objects
  configs : List GraphLoopsMemorySCF.Configuration
  module : Module
  opcodes : Opcodes
  specifications : Specifications
  mapping : Aliases
  views : List PrivateView
  callers : List CallerView
  deriving DecidableEq, Repr

inductive Command where
  | constant (key : String) (result : Binding) (value : Nat)
  | binary (key operation : String) (left right result : Binding) (element : String)
  | allocate (operation : String) (item : PrivateView) (retained : Bool)
  | release (operation : String) (item : PrivateView) (retained : Bool)
  | initialize (link : InitializerLink)
  | load (key operation : String) (access : Access) (result : Binding)
  | store (key operation : String) (access : Access) (operand : Binding)
  | observer (key operation name : String) (input buffer result : Binding) (program : Touch)
  deriving DecidableEq, Repr

def Command.atom : Command → Atom
  | .constant _ result value => .constant result value
  | .binary _ operation left right result _ => .action operation [left, right] [result]
  | .allocate operation item _ => .action operation [] [item.binding]
  | .release operation item _ => .action operation [item.binding] []
  | .initialize link => link.canonical.atom
  | .load _ operation access result => .action operation (access.buffer :: access.indices) [result]
  | .store _ operation access operand => .action operation (operand :: access.buffer :: access.indices) []
  | .observer _ operation _ input buffer result _ => .action operation [input, buffer] [result]

def Command.instruction : Command → GraphLoopsReuse.Instruction
  | .constant key _ _ | .binary key _ _ _ _ _ | .observer key _ _ _ _ _ _ => .retained key
  | .allocate _ item retained => .allocate item.object retained
  | .release _ item retained => .release item.object retained
  | .initialize link => .initialize link.config.object link.config.key
  | .load key _ access _ => .load access.object key
  | .store key _ access _ => .store access.object key

def Command.results : Command → List Binding
  | .constant _ result _ | .binary _ _ _ _ result _ | .load _ _ _ result | .observer _ _ _ _ _ result _ => [result]
  | .allocate _ item _ => [item.binding]
  | _ => []

def Command.inputs : Command → List Binding
  | .constant _ _ _ | .allocate _ _ _ => []
  | .binary _ _ left right _ _ => [left, right]
  | .release _ item _ => [item.binding]
  | .initialize link => [link.canonical.seed, link.canonical.buffer, link.canonical.zero,
      link.canonical.rows, link.canonical.columns, link.canonical.one]
  | .load _ _ access _ => access.buffer :: access.indices
  | .store _ _ access operand => operand :: access.buffer :: access.indices
  | .observer _ _ _ input buffer _ _ => [input, buffer]

def Command.nextClock (clock : Nat) : Command → Nat
  | .allocate _ _ _ => clock + 1
  | _ => clock

def ScalarBinding (binding : Binding) (element : String) : Prop :=
  binding.kind = .scalar element ∧ binding.known = none

instance scalarBindingDecidable (binding : Binding) (element : String) : Decidable (ScalarBinding binding element) :=
  inferInstanceAs (Decidable (_ ∧ _))

def PrivateView.Valid (plan : PlanContext) (item : PrivateView) : Prop :=
  item ∈ plan.views ∧ item.object < plan.objects.length ∧
  item.shape = (GraphLoopsReuse.objectAt plan.objects item.object).shape ∧
  item.binding.kind = .view (type item.shape item.element) ∧ item.binding.known = none

instance privateViewValidDecidable (plan : PlanContext) (item : PrivateView) : Decidable (item.Valid plan) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def Access.Valid (plan : PlanContext) (access : Access) : Prop :=
  (PrivateView.mk access.buffer access.object access.shape access.element).Valid plan ∧
    ∀ index ∈ access.indices, index.kind = .index

instance accessValidDecidable (plan : PlanContext) (access : Access) : Decidable (access.Valid plan) :=
  inferInstanceAs (Decidable (_ ∧ _))

def Command.Operations (plan : PlanContext) (context : Context) (clock : Nat) : Command → Prop
  | .constant key result value =>
      specAt plan.specifications key = some (.constant result value) ∧ result.kind = .index ∧ result.known = some value
  | .binary key operation left right result element =>
      specAt plan.specifications key = some (.binary operation left right result element) ∧
      opcodeAt plan.opcodes operation = some (.binary element) ∧ callTarget plan.module.calls operation = none ∧
      ScalarBinding left element ∧ ScalarBinding right element ∧ ScalarBinding result element
  | .allocate operation item _ =>
      item.Valid plan ∧ item.object = clock ∧ opcodeAt plan.opcodes operation = some (.allocate item.shape item.element) ∧
      callTarget plan.module.calls operation = none
  | .release operation item _ =>
      item.Valid plan ∧ opcodeAt plan.opcodes operation = some .release ∧ callTarget plan.module.calls operation = none
  | .initialize link =>
      link.Facts ∧ link.Bound context plan.views ∧ link.config ∈ plan.configs ∧ link.mapping = plan.mapping ∧
      link.config.object < plan.objects.length ∧ opcodeAt plan.opcodes link.canonical.storeKey = some .store ∧
      opcodeAt plan.opcodes link.canonical.loopKey = some .loop
  | .load key operation access result =>
      specAt plan.specifications key = some (.load access result) ∧ access.Valid plan ∧
      opcodeAt plan.opcodes operation = some .load ∧ callTarget plan.module.calls operation = none ∧
      ScalarBinding result access.element
  | .store key operation access operand =>
      specAt plan.specifications key = some (.store access operand) ∧ access.Valid plan ∧
      opcodeAt plan.opcodes operation = some .store ∧ callTarget plan.module.calls operation = none ∧
      ScalarBinding operand access.element
  | .observer key operation name input buffer result program =>
      specAt plan.specifications key = some (.observer program.binaryKey input buffer result program.element) ∧
      program.Facts ∧ functionAt plan.module name = some (program.function name) ∧
      program.input.name ≠ program.buffer.name ∧ callTarget plan.module.calls operation = some name ∧
      callTarget plan.module.calls program.loadKey = none ∧ callTarget plan.module.calls program.binaryKey = none ∧
      callTarget plan.module.calls program.storeKey = none ∧
      opcodeAt plan.opcodes program.loadKey = some .load ∧ opcodeAt plan.opcodes program.storeKey = some .store ∧
      opcodeAt plan.opcodes program.binaryKey = some (.binary program.element) ∧
      ScalarBinding input program.element ∧ ScalarBinding result program.element ∧
      CallerView.mk buffer program.element ∈ plan.callers

instance commandOperationsDecidable (plan : PlanContext) (context : Context) (clock : Nat) (command : Command) :
    Decidable (command.Operations plan context clock) := by
  cases command <;> unfold Command.Operations <;> infer_instance

def Command.Protects (plan : PlanContext) (command : Command) : Prop :=
  match command with
  | .allocate _ _ _ => True
  | _ => ∀ item ∈ plan.views, item.binding.name ∉ names command.results

instance commandProtectsDecidable (plan : PlanContext) (command : Command) : Decidable (command.Protects plan) := by
  cases command <;> unfold Command.Protects <;> infer_instance

def Command.Valid (plan : PlanContext) (context : Context) (clock : Nat) (command : Command) : Prop :=
  command.Operations plan context clock ∧ AllBound context command.inputs ∧ Fresh context command.results ∧
    command.Protects plan ∧ CallersBound context plan.callers

instance commandValidDecidable (plan : PlanContext) (context : Context) (clock : Nat) (command : Command) :
    Decidable (command.Valid plan context clock) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def CommandsValid (plan : PlanContext) : Context → Nat → List Command → Prop
  | _, _, [] => True
  | context, clock, command :: rest => command.Valid plan context clock ∧
      CommandsValid plan (extendContext context command.results) (command.nextClock clock) rest

def commandsDecidable (plan : PlanContext) : (context : Context) → (clock : Nat) → (commands : List Command) →
    Decidable (CommandsValid plan context clock commands)
  | _, _, [] => isTrue True.intro
  | context, clock, command :: rest =>
      letI := commandsDecidable plan (extendContext context command.results) (command.nextClock clock) rest
      inferInstanceAs (Decidable (_ ∧ _))

instance commandsValidDecidable (plan : PlanContext) (context : Context) (clock : Nat) (commands : List Command) :
    Decidable (CommandsValid plan context clock commands) := commandsDecidable plan context clock commands

end Merlin.GraphLoopsShape.Epoch
