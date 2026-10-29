/- UNCOMPILED. Finite expansion plan for the actual retained memory case.
Every expansion records its EXACT old command; no instruction is omitted from
source binding. These checks are not a substitute for per-command simulation.
-/
import Merlin.GraphLoopsMemorySCFAfterScope
import Merlin.GraphLoopsShapeEpochFunction
import Merlin.GraphLoopsShapeAfter

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

inductive Expansion where
  | keep (command : Epoch.Command)
  | alias (operation : String) (item owner : Epoch.PrivateView)
  | dropRelease (operation : String) (item : Epoch.PrivateView)
  | tiled (link : Epoch.InitializerLink) (program : TiledInitializer)
  deriving DecidableEq, Repr

def Expansion.source : Expansion → Epoch.Command
  | .keep command => command
  | .alias operation item _ => .allocate operation item false
  | .dropRelease operation item => .release operation item false
  | .tiled link _ => .initialize link

def Expansion.fragments : Expansion → List Fragment
  | .keep command => [.atom command.atom]
  | .alias _ item owner => [.alias owner.binding item.binding (Epoch.type item.shape item.element)]
  | .dropRelease _ _ => [.empty]
  | .tiled link program =>
      [.atom (.constant program.tile link.config.tile),
       .atom (.region program.uniform.loopKey [program.uniform.zero, program.uniform.rows, program.tile]
         [program.tileIV] [] program.body)]

def Expansion.nextClock (clock : Nat) (expansion : Expansion) : Nat :=
  match expansion.source with
  | .allocate _ _ true => clock + 1
  | _ => clock

def Expansion.generated : Expansion → List Binding
  | .tiled _ program => [program.tile, program.tileIV, program.sum, program.bound]
  | _ => []

def Expansion.Facts (bits : Nat) (plan : Epoch.PlanContext) (indices : IndexOpcodes)
    (context : Context) (clock : Nat) : Expansion → Prop
  | .keep command => match command with
    | .allocate _ item true => item.Valid plan ∧
        GraphLoopsReuse.slotOf plan.objects item.object = clock ∧ clock < plan.objects.length
    | .release _ item true => item.Valid plan ∧ GraphLoopsReuse.slotOf plan.objects item.object < plan.objects.length
    | .allocate _ _ false | .release _ _ false | .initialize _ => False
    | _ => True
  | .alias _ item owner =>
      item.Valid plan ∧ owner.Valid plan ∧ owner.binding ∈ context ∧
      item.shape = owner.shape ∧ item.element = owner.element ∧
      GraphLoopsReuse.slotOf plan.objects item.object = GraphLoopsReuse.slotOf plan.objects owner.object ∧
      GraphLoopsReuse.slotOf plan.objects item.object < plan.objects.length
  | .dropRelease _ item => item.Valid plan
  | .tiled link program =>
      link.Facts ∧ program.uniform = link.canonical ∧
      program.tile.kind = .index ∧ program.tile.known = some link.config.tile ∧
      program.ControlFacts ∧ (GraphLoopsSegment.Bounds.mk bits 0 link.rows 1 link.config.tile).Valid ∧
      (∀ binding ∈ [program.tile, program.tileIV, program.sum, program.bound], binding.name ∉ names context) ∧
      indexOpcodeAt indices program.addKey = some .add ∧ indexOpcodeAt indices program.minKey = some .minimum ∧
      Epoch.opcodeAt plan.opcodes program.addKey = none ∧ Epoch.opcodeAt plan.opcodes program.minKey = none ∧
      callTarget plan.module.calls program.addKey = none ∧ callTarget plan.module.calls program.minKey = none

instance expansionFactsDecidable (bits : Nat) (plan : Epoch.PlanContext) (indices : IndexOpcodes)
    (context : Context) (clock : Nat) (expansion : Expansion) :
    Decidable (expansion.Facts bits plan indices context clock) := by
  cases expansion with
  | keep command =>
    cases command with
    | allocate operation item retained => cases retained <;> unfold Expansion.Facts <;> infer_instance
    | release operation item retained => cases retained <;> unfold Expansion.Facts <;> infer_instance
    | constant | binary | «initialize» | load | store | observer => unfold Expansion.Facts; infer_instance
  | alias | dropRelease | tiled => unfold Expansion.Facts; infer_instance

def ExpansionsValid (bits : Nat) (plan : Epoch.PlanContext) (indices : IndexOpcodes) :
    Context → Nat → List Expansion → Prop
  | _, _, [] => True
  | context, clock, expansion :: rest => expansion.Facts bits plan indices context clock ∧
      ExpansionsValid bits plan indices (extendContext context expansion.source.results)
        (expansion.nextClock clock) rest

def expansionsDecidable (bits : Nat) (plan : Epoch.PlanContext) (indices : IndexOpcodes) :
    (context : Context) → (clock : Nat) → (expansions : List Expansion) →
      Decidable (ExpansionsValid bits plan indices context clock expansions)
  | _, _, [] => isTrue trivial
  | context, clock, expansion :: rest =>
      letI := expansionsDecidable bits plan indices (extendContext context expansion.source.results)
        (expansion.nextClock clock) rest
      inferInstanceAs (Decidable (_ ∧ _))

instance (bits : Nat) (plan : Epoch.PlanContext) (indices : IndexOpcodes) (context : Context)
    (clock : Nat) (expansions : List Expansion) :
    Decidable (ExpansionsValid bits plan indices context clock expansions) :=
  expansionsDecidable bits plan indices context clock expansions

/-- Adding generated index meanings may not override ANY existing primitive
or call key. The same scalar/caller interpretation is therefore still available.
-/
def NoOpcodeOverrides (plan : Epoch.PlanContext) (indices : IndexOpcodes) : Prop :=
  (∀ pair ∈ plan.opcodes, indexOpcodeAt indices pair.1 = none) ∧
    (∀ pair ∈ plan.module.calls, indexOpcodeAt indices pair.1 = none)

instance (plan : Epoch.PlanContext) (indices : IndexOpcodes) : Decidable (NoOpcodeOverrides plan indices) := by
  unfold NoOpcodeOverrides
  infer_instance

def expandedModule (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) : Module :=
  { plan.module with functions := plan.module.functions.map (fun function =>
      if function.name = kernel.function.name then
        { function with code := fragmentsCode (expansions.flatMap Expansion.fragments) kernel.returned }
      else function) }

def ExpansionBinding (bits : Nat) (plan : Epoch.PlanContext) (kernel : Epoch.Kernel)
    (indices : IndexOpcodes) (expansions : List Expansion) : Prop :=
  expansions.map Expansion.source = kernel.commands ∧
    ExpansionsValid bits plan indices kernel.function.parameters 0 expansions ∧
    NoOpcodeOverrides plan indices ∧ NoViewBindings kernel.returned ∧
    (expandedModule plan kernel expansions).Valid bits

instance (bits : Nat) (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (indices : IndexOpcodes)
    (expansions : List Expansion) : Decidable (ExpansionBinding bits plan kernel indices expansions) := by
  unfold ExpansionBinding
  infer_instance

theorem expansion_source_exact (bits : Nat) (plan : Epoch.PlanContext) (kernel : Epoch.Kernel)
    (indices : IndexOpcodes) (expansions : List Expansion)
    (checked : decide (ExpansionBinding bits plan kernel indices expansions) = true) :
    expansions.map Expansion.source = kernel.commands := (of_decide_eq_true checked).1

theorem expanded_calls (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) :
    (expandedModule plan kernel expansions).calls = plan.module.calls := rfl

/-- The terminal materialization edge is an ordinary complete-function theorem.
Its source is the computed EXPANDED module, not the old source kernel. The
source→expanded proof must still be supplied by local operational simulation.
-/
theorem checked_materialization (bits : Nat) (plan : Epoch.PlanContext) (kernel : Epoch.Kernel)
    (expansions : List Expansion) (actualFinal : Module) (swaps : List (Nat × Nat))
    (sourceFrame finalFrame : List String)
    (checked : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) (name : String)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning (expandedModule plan kernel expansions) false name inputs before outputs after ↔
      complete meaning actualFinal false name inputs before outputs after :=
  GraphLoopsShape.After.checked_complete checked meaning name inputs before outputs after

end Merlin.GraphLoopsMemorySCF.After
