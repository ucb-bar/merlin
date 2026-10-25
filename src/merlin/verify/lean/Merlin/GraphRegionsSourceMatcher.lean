/- Executable complete-source matcher. All actual AST fields, SSA identities,
types, nested scalar bodies, iteration maps and returned order are recomputed.
Certificates contain descriptions, never trusted eligibility/success Booleans. -/
import Merlin.GraphRegionsSourceModel

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Sem.Ops

private def valueDecEq (first second : Value) : Decidable (first = second) :=
  decidable_of_iff (first.id = second.id ∧ first.ty = second.ty)
    (by cases first; cases second; simp)
local instance : DecidableEq Value := valueDecEq

def headMatches (name : String) (results : List Value) (operands : List Nat)
    (attrs : List (String × Attr)) (source : Op) : Bool :=
  decide (source.name = name ∧ source.results = results ∧ source.operands = operands ∧
    source.attrs = attrs ∧ source.props = [] ∧ source.successors = [])

theorem headMatches_sound {name : String} {results : List Value} {operands : List Nat}
    {attrs : List (String × Attr)} {source : Op}
    (accepted : headMatches name results operands attrs source = true) :
    source = .mk name results operands attrs [] source.regions [] := by
  have parsed : source.name = name ∧ source.results = results ∧ source.operands = operands ∧
      source.attrs = attrs ∧ source.props = [] ∧ source.successors = [] := of_decide_eq_true accepted
  cases source with
  | mk actual rs args attributes properties regions successors =>
    obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := parsed
    rfl

def terminalMatches (name : String) (operands : List Nat) (source : Op) : Bool :=
  headMatches name [] operands [] source && decide source.regions.isEmpty

theorem terminalMatches_sound {name : String} {operands : List Nat} {source : Op}
    (accepted : terminalMatches name operands source = true) : source = .mk name [] operands [] [] [] [] := by
  simp only [terminalMatches, Bool.and_eq_true, decide_eq_true_eq] at accepted
  have empty : source.regions = [] := List.isEmpty_iff.mp accepted.2
  rw [headMatches_sound accepted.1, empty]

def scalarBodyMatches (returned : Nat) : List Instruction → List Op → Bool
  | [], [source] => terminalMatches "linalg.yield" [returned] source
  | instruction :: rest, source :: tail => instruction.matches source && scalarBodyMatches returned rest tail
  | _, _ => false

theorem scalarBodyMatches_sound {returned : Nat} {instructions : List Instruction} {source : List Op}
    (accepted : scalarBodyMatches returned instructions source = true) :
    source = instructions.map Instruction.operation ++ [yieldOp returned] := by
  induction instructions generalizing source with
  | nil =>
    cases source with
    | nil => simp [scalarBodyMatches] at accepted
    | cons op tail =>
      cases tail with
      | nil => rw [terminalMatches_sound accepted]; rfl
      | cons _ _ => simp [scalarBodyMatches] at accepted
  | cons instruction rest ih =>
    cases source with
    | nil => simp [scalarBodyMatches] at accepted
    | cons op tail =>
      simp only [scalarBodyMatches, Bool.and_eq_true] at accepted
      rw [instruction.matches_sound accepted.1, ih accepted.2]
      rfl

def bodyMatches (body : Body) : Region → Bool
  | .mk [.mk arguments operations] =>
    decide (arguments = inputArguments body.inputs ++ [⟨body.old, .int body.width⟩]) &&
      scalarBodyMatches body.returned body.instructions operations
  | _ => false

theorem bodyMatches_sound {body : Body} {source : Region} (accepted : bodyMatches body source = true) :
    source = body.region := by
  unfold bodyMatches at accepted
  split at accepted
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at accepted
    rw [accepted.1, scalarBodyMatches_sound accepted.2]
    rfl
  · contradiction

def stageMatches (stage : Stage) (source : Op) : Bool :=
  headMatches "linalg.generic" [⟨stage.result, stage.output.ty⟩] (stage.inputs ++ [stage.destination])
    stage.attrs source &&
      (match source.regions with | [region] => bodyMatches stage.body region | _ => false)

theorem stageMatches_sound {stage : Stage} {source : Op} (accepted : stageMatches stage source = true) :
    source = stage.operation := by
  simp only [stageMatches, Bool.and_eq_true] at accepted
  have head := headMatches_sound accepted.1
  split at accepted
  · next region regions =>
    rw [head, regions, bodyMatches_sound accepted.2]
    rfl
  · simp at accepted

def itemMatches (item : Item) (source : Op) : Bool :=
  match item with
  | .empty id type => headMatches "tensor.empty" [⟨id, type.ty⟩] [] [] source && decide source.regions.isEmpty
  | .constant value => (Instruction.mk (.constant value.width value.literal) value.id []).matches source
  | .generic entry => stageMatches entry.stage source

theorem itemMatches_sound {item : Item} {source : Op} (accepted : itemMatches item source = true) :
    source = item.operation := by
  cases item with
  | empty id type =>
    simp only [itemMatches, Bool.and_eq_true, decide_eq_true_eq] at accepted
    have empty : source.regions = [] := List.isEmpty_iff.mp accepted.2
    rw [headMatches_sound accepted.1, empty]
    rfl
  | generic entry => exact stageMatches_sound accepted
  | constant value => exact Instruction.matches_sound accepted

def itemsMatch (returned : List Nat) : List Item → List Op → Bool
  | [], [source] => terminalMatches "func.return" returned source
  | item :: rest, source :: tail => itemMatches item source && itemsMatch returned rest tail
  | _, _ => false

theorem itemsMatch_sound {returned : List Nat} {items : List Item} {source : List Op}
    (accepted : itemsMatch returned items source = true) :
    source = items.map Item.operation ++ [returnOperation returned] := by
  induction items generalizing source with
  | nil =>
    cases source with
    | nil => simp [itemsMatch] at accepted
    | cons op tail =>
      cases tail with
      | nil => rw [terminalMatches_sound accepted]; rfl
      | cons _ _ => simp [itemsMatch] at accepted
  | cons item rest ih =>
    cases source with
    | nil => simp [itemsMatch] at accepted
    | cons op tail =>
      simp only [itemsMatch, Bool.and_eq_true] at accepted
      rw [itemMatches_sound accepted.1, ih accepted.2]
      rfl

def functionBodyMatches (program : Program) : Region → Bool
  | .mk [.mk arguments operations] =>
    decide (arguments = TensorArguments program.arguments) && itemsMatch program.returned program.items operations
  | _ => false

theorem functionBodyMatches_sound {program : Program} {source : Region}
    (accepted : functionBodyMatches program source = true) : source = program.body := by
  unfold functionBodyMatches at accepted
  split at accepted
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at accepted
    rw [accepted.1, itemsMatch_sound accepted.2]
    rfl
  · contradiction

def functionMatches (program : Program) (source : Op) : Bool :=
  headMatches "func.func" [] [] program.attrs source &&
    (match source.regions with | [region] => functionBodyMatches program region | _ => false)

theorem functionMatches_sound {program : Program} {source : Op}
    (accepted : functionMatches program source = true) : source = program.operation := by
  simp only [functionMatches, Bool.and_eq_true] at accepted
  have head := headMatches_sound accepted.1
  split at accepted
  · next region regions =>
    rw [head, regions, functionBodyMatches_sound accepted.2]
    rfl
  · simp at accepted

def Program.matches (program : Program) (source : Module) : Bool :=
  decide (source.attrs = [] ∧ source.props = []) &&
    (match source.ops with | [function] => functionMatches program function | _ => false)

theorem Program.matches_sound {program : Program} {source : Module}
    (accepted : program.matches source = true) : source = program.module := by
  simp only [Program.matches, Bool.and_eq_true, decide_eq_true_eq] at accepted
  have attributes := accepted.1
  split at accepted
  · next function operations =>
    have equal := functionMatches_sound accepted.2
    cases source
    simp_all [Program.module]
  · simp at accepted

/-- Completeness avoids evaluating the frozen, opaque attribute Boolean equality.
The kernel still checks exact equality of every original AST field. -/
theorem Instruction.matches_self (instruction : Instruction) :
    instruction.matches instruction.operation = true := by
  simp [Instruction.matches, Instruction.operation, Code.operation]

theorem scalarBodyMatches_self (returned : Nat) (instructions : List Instruction) :
    scalarBodyMatches returned instructions
      (instructions.map Instruction.operation ++ [yieldOp returned]) = true := by
  induction instructions with
  | nil => simp [scalarBodyMatches, terminalMatches, headMatches, yieldOp,
      Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, Op.regions]
  | cons instruction rest ih =>
    simpa [scalarBodyMatches, Instruction.matches_self] using ih

theorem bodyMatches_self (body : Body) : bodyMatches body body.region = true := by
  simp [bodyMatches, Body.region, scalarBodyMatches_self]

theorem stageMatches_self (stage : Stage) : stageMatches stage stage.operation = true := by
  simp [stageMatches, Stage.operation, headMatches, bodyMatches_self,
    Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, Op.regions]

theorem itemMatches_self (item : Item) : itemMatches item item.operation = true := by
  cases item with
  | empty id type => simp [itemMatches, Item.operation, emptyOperation, headMatches,
      Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, Op.regions]
  | constant value =>
    simpa only [itemMatches, Item.operation, CapturedConstant.operation, Instruction.operation] using
      Instruction.matches_self ⟨.constant value.width value.literal, value.id, []⟩
  | generic entry => exact stageMatches_self _

theorem itemsMatch_self (returned : List Nat) (items : List Item) :
    itemsMatch returned items (items.map Item.operation ++ [returnOperation returned]) = true := by
  induction items with
  | nil => simp [itemsMatch, terminalMatches, headMatches, returnOperation,
      Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, Op.regions]
  | cons item rest ih => simpa [itemsMatch, itemMatches_self] using ih

theorem Program.matches_self (program : Program) : program.matches program.module = true := by
  simp [Program.matches, Program.module, functionMatches, Program.operation, headMatches,
    functionBodyMatches, Program.body, itemsMatch_self,
    Op.name, Op.results, Op.operands, Op.attrs, Op.props, Op.successors, Op.regions]

theorem Program.matches_iff {program : Program} {source : Module} :
    program.matches source = true ↔ source = program.module :=
  ⟨Program.matches_sound, fun equal => equal ▸ program.matches_self⟩

instance (type : TensorType) : Decidable type.Valid := by unfold TensorType.Valid; infer_instance
instance (stage : Stage) : Decidable stage.Valid := by unfold Stage.Valid; infer_instance
instance (stage : Stage) (types : TensorTypes) : Decidable (stage.InputsTyped types) := by
  unfold Stage.InputsTyped
  infer_instance
instance (item : Item) (entries : List Entry) (types : Nat → TensorType) (logical initializers : TensorTypes) :
    (constants : List CapturedConstant) → Decidable (item.Valid entries types logical initializers constants) := by
  intro constants
  cases item <;> unfold Item.Valid CapturesAvailable <;> infer_instance

def itemsValidDec (entries : List Entry) (types : Nat → TensorType) :
    (logical initializers : TensorTypes) → (constants : List CapturedConstant) → (items : List Item) →
      Decidable (ItemsValid entries types logical initializers constants items)
  | _, _, _, [] => isTrue trivial
  | logical, initializers, constants, item :: rest =>
    letI := itemsValidDec entries types (item.nextLogical logical) (item.nextInitializers initializers)
      (item.nextConstants constants) rest
    inferInstanceAs (Decidable (item.Valid entries types logical initializers constants ∧ ItemsValid entries types _ _ _ rest))
instance (entries : List Entry) (types : Nat → TensorType) (logical initializers : TensorTypes)
    (constants : List CapturedConstant) (items : List Item) :
    Decidable (ItemsValid entries types logical initializers constants items) :=
  itemsValidDec entries types logical initializers constants items

def ordinalsDec (ordinal : Nat) : (items : List Item) → Decidable (Ordinals ordinal items)
  | [] => isTrue trivial
  | .empty _ _ :: rest => ordinalsDec (ordinal + 1) rest
  | .constant _ :: rest => ordinalsDec (ordinal + 1) rest
  | .generic entry :: rest =>
    letI := ordinalsDec (ordinal + entry.stage.body.instructions.length + 2) rest
    inferInstanceAs (Decidable (entry.stage.ordinal = ordinal ∧ Ordinals _ rest))
instance (ordinal : Nat) (items : List Item) : Decidable (Ordinals ordinal items) := ordinalsDec ordinal items

instance (program : Program) : Decidable program.Valid := by
  unfold Program.Valid InputTypesBound ReturnedTyped
  infer_instance

def Program.graphMatches (program : Program) (graph : GraphRegions.Graph) : Bool :=
  decide (graph.nodes = graphNodes program.items ∧ graph.inputs = program.arguments.map Prod.fst ∧
    graph.observations = program.returned)

def checkSource (before after : Module) (program : Program) (graph : GraphRegions.Graph) : Bool :=
  program.matches before && program.matches after && decide program.Valid && program.graphMatches graph

theorem checkSource_iff {before after : Module} {program : Program} {graph : GraphRegions.Graph} :
    checkSource before after program graph = true ↔
      ((before = program.module ∧ after = program.module) ∧ program.Valid) ∧
        program.graphMatches graph = true := by
  simp only [checkSource, Bool.and_eq_true, Program.matches_iff, decide_eq_true_eq]

/-- Acceptance connects complete actual source ASTs to the constructed operational
model, and proves source success/initialized outputs on the input-only domain. -/
theorem checkSource_sound {before after : Module} {program : Program} {graph : GraphRegions.Graph}
    (accepted : checkSource before after program graph = true) (values : List Tensor)
    (domain : InputsDomain program.arguments values) (ctx : Ctx) :
    before = after ∧
      MerlinVerify.denote opSem ctx before program.name (values.map RVal.tensor) =
        .ok ((graph.observations.map
          (GraphRegions.executeNodes (sourceMeaning program.entries program.types) graph.nodes
            (entryState program.types program.arguments values))).map RVal.tensor) ∧
      ResultsDefined ((graph.observations.map
          (GraphRegions.executeNodes (sourceMeaning program.entries program.types) graph.nodes
            (entryState program.types program.arguments values))).map RVal.tensor) := by
  simp only [checkSource, Bool.and_eq_true, decide_eq_true_eq, Program.graphMatches] at accepted
  obtain ⟨⟨⟨original, retained⟩, valid⟩, nodes, _, observations⟩ := accepted
  rw [Program.matches_sound original, Program.matches_sound retained, nodes, observations]
  exact ⟨rfl, program_denote program valid values domain ctx⟩

/-- Full actual-source to checked logical region theorem. Concrete native
implementation/alias/completion contracts remain separate and are not inferred. -/
theorem checkSource_program {before after : Module} {program : Program} {graph retained : GraphRegions.Graph}
    (sourceAccepted : checkSource before after program graph = true)
    {start count : Nat} {selection : List Nat} {plan : GraphRegions.Plan} {fused : GraphRegions.FusedBody}
    (graphAccepted : GraphRegions.checkRealization graph retained start count selection plan fused = true)
    (values : List Tensor) (domain : InputsDomain program.arguments values) (ctx : Ctx) :
    MerlinVerify.denote opSem ctx before program.name (values.map RVal.tensor) =
      .ok ((graph.observations.map
        (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (graph.nodes.drop (start + count))
          (fused.execute (sourceMeaning program.entries program.types)
            (GraphRegions.executeNodes (sourceMeaning program.entries program.types) (graph.nodes.take start)
              (entryState program.types program.arguments values))))).map RVal.tensor) := by
  simp only [checkSource, Bool.and_eq_true, decide_eq_true_eq, Program.graphMatches] at sourceAccepted
  obtain ⟨⟨⟨original, _⟩, valid⟩, nodes, _, observations⟩ := sourceAccepted
  rw [Program.matches_sound original]
  exact program_checked program valid values domain ctx nodes observations graphAccepted

end Merlin.GraphRegionsSource
