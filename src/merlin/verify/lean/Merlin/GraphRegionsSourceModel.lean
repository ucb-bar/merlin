/- Complete closed source-function model. All finite static facts are checked;
the only dynamic domain is initialized tensor arguments of the actual signature.
Source execution, type preservation and initialized observations are derived. -/
import Merlin.GraphRegionsSourceInputs

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def Item.entries : Item → List Entry
  | .empty .. => []
  | .constant _ => []
  | .generic entry => [entry]

def Item.definitions : Item → TensorTypes
  | .empty id type => [(id, type)]
  | .constant _ => []
  | .generic entry => [(entry.stage.result, entry.stage.output)]

def Item.allIds : Item → List Nat
  | .empty id _ => [id]
  | .constant value => [value.id]
  | .generic entry => [entry.stage.result] ++ entry.stage.body.inputs.map Prod.fst ++
      [entry.stage.body.old] ++ entry.stage.body.instructions.map Instruction.result

def Item.treeSize : Item → Nat
  | .empty .. => 1
  | .constant _ => 1
  | .generic entry => entry.stage.body.instructions.length + 2

def Ordinals : Nat → List Item → Prop
  | _, [] => True
  | ordinal, .empty _ _ :: rest => Ordinals (ordinal + 1) rest
  | ordinal, .constant _ :: rest => Ordinals (ordinal + 1) rest
  | ordinal, .generic entry :: rest => entry.stage.ordinal = ordinal ∧
      Ordinals (ordinal + entry.stage.body.instructions.length + 2) rest

structure Program where
  name : String
  arguments : TensorTypes
  items : List Item
  returned : List Nat
  results : List TensorType
  deriving DecidableEq, Repr

def Program.entries (program : Program) : List Entry := program.items.flatMap Item.entries
def Program.definitions (program : Program) : TensorTypes :=
  program.arguments ++ program.items.flatMap Item.definitions
def Program.types (program : Program) (id : Nat) : TensorType :=
  (program.definitions.lookup id).getD ⟨[], 8⟩

def Program.body (program : Program) : Region :=
  .mk [.mk (TensorArguments program.arguments)
    (program.items.map Item.operation ++ [returnOperation program.returned])]

def Program.attrs (program : Program) : List (String × Attr) :=
  [("function_type", .type (.function (program.arguments.map fun pair => pair.2.ty)
      (program.results.map TensorType.ty))), ("sym_name", .str program.name)]

def Program.operation (program : Program) : Op :=
  .mk "func.func" [] [] program.attrs [] [program.body] []

def Program.module (program : Program) : Module := ⟨[program.operation], [], []⟩

def Program.Valid (program : Program) : Prop :=
  (program.arguments.map Prod.fst ++ program.items.flatMap Item.allIds).Nodup ∧
    (∀ argument ∈ program.arguments, argument.2.Valid) ∧
    InputTypesBound program.types program.arguments ∧
    ItemsValid program.entries program.types program.arguments [] [] program.items ∧
    ReturnedTyped (finalLogical program.items program.arguments) program.returned program.results ∧
    Ordinals 2 program.items

def Program.func (program : Program) : Func := ⟨program.name, program.body, false, false⟩

theorem program_functions (program : Program) : program.module.funcs = [program.func] := by
  simp [Program.module, Program.operation, Program.attrs, Program.func, Program.body,
    Module.funcs, Op.asFunc?, Op.name, Op.attr?, Op.props, Op.attrs, Op.regions, List.lookup,
    Region.blocks]

theorem program_function (program : Program) : program.module.func? program.name = some program.func := by
  simp [Module.func?, program_functions, Program.func]

/-- Actual frozen function execution, including exact acyclic call fuel. The
domain states only the actual initialized input tensors and their signature. -/
theorem program_denote (program : Program) (valid : program.Valid) (values : List Tensor)
    (domain : InputsDomain program.arguments values) (ctx : Ctx) :
    denote opSem ctx program.module program.name (values.map RVal.tensor) =
        .ok ((program.returned.map
          (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (graphNodes program.items)
            (entryState program.types program.arguments values))).map RVal.tensor) ∧
      ResultsDefined ((program.returned.map
          (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (graphNodes program.items)
            (entryState program.types program.arguments values))).map RVal.tensor) := by
  have emptyPool : PoisonRel [] (entryEnvironment program.arguments values) := by
    intro id type found
    simp [List.lookup] at found
  have emptyConstants : ConstantsRel [] (entryEnvironment program.arguments values) := by
    intro constant member
    simp at member
  have source := source_observers valid.2.2.2.1 (entry_related domain program.types)
    emptyPool emptyConstants valid.2.2.2.2.1 ctx (evalFunc opSem ctx program.module 1)
  refine ⟨?_, source.2⟩
  unfold MerlinVerify.denote
  rw [program_functions]
  change evalFunc opSem ctx program.module 2 program.name (values.map RVal.tensor) = _
  rw [evalFunc, program_function]
  change evalRegion opSem ctx (evalFunc opSem ctx program.module 1) [] program.body (values.map RVal.tensor) = _
  simp only [Program.body, evalRegion, evalBlocks, evalBlock]
  rw [bind_function_inputs domain, Outcome.bind_ok]
  exact source.1

/-- Acceptance of the earlier actual graph checker now preserves the frozen
source function's full returned list under the constructed integer interpretation.
No source-success/global-typing premise is added to the dynamic input domain. -/
theorem program_checked (program : Program) (valid : program.Valid) (values : List Tensor)
    (domain : InputsDomain program.arguments values) (ctx : Ctx)
    {before after : GraphRegions.Graph} {start count : Nat} {selection : List Nat}
    {plan : GraphRegions.Plan} {fused : GraphRegions.FusedBody}
    (nodes : before.nodes = graphNodes program.items) (observations : before.observations = program.returned)
    (accepted : GraphRegions.checkRealization before after start count selection plan fused = true) :
    denote opSem ctx program.module program.name (values.map RVal.tensor) =
      .ok ((before.observations.map
        (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (before.nodes.drop (start + count))
          (fused.execute (sourceMeaning program.entries program.types)
            (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (before.nodes.take start)
              (entryState program.types program.arguments values))))).map RVal.tensor) := by
  have source := (program_denote program valid values domain ctx).1
  have typed := entry_typed domain program.types valid.2.2.1
  have observers := (GraphRegions.checkRealization_sound accepted
    (sourceMeaning program.entries program.types) _ typed).2.2
  rw [nodes, observations] at observers
  rw [source, nodes, observations, observers]

end Merlin.GraphRegionsSource
