/- A complete narrow parent grammar for
captured-literal extraction: one tensor function, a checked literal/static-empty
prefix, one captured parallel generic and its sole tensor return. Extra helpers,
calls, regions, effects, dynamic shapes or returned values are outside this
candidate. This does not alter any frozen source semantics or qualification. -/
import Merlin.GraphRegionsDeclaredCaptureTensor

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegionsDeclared

structure Parent where
  name : String
  visibility : Option String
  site : GenericSite
  references : List Reference
  leading : List Prefix.Instruction
  deriving DecidableEq, Repr

def Parent.tensorType (parent : Parent) : Ty :=
  .tensor (parent.site.shape.map some) (.int parent.site.stage.program.output) none

def Parent.module (parent : Parent) : Module :=
  let count := parent.site.stage.program.inputs.length
  let inputs := (List.range count).map fun id => (⟨id, parent.tensorType⟩ : Value)
  let attrs := [("function_type", .type (.function (List.replicate count parent.tensorType) [parent.tensorType])),
    ("sym_name", .str parent.name)] ++ parent.visibility.toList.map (fun value => ("sym_visibility", .str value))
  ⟨[.mk "func.func" [] [] attrs [] [.mk [.mk inputs
    (parent.leading.map Prefix.Instruction.emit ++ [originalGeneric parent.site parent.references,
      .mk "func.return" [] [parent.site.stage.output] [] [] [] []])]] []], [], []⟩

def Parent.Valid (parent : Parent) : Prop :=
  parent.site.stage.ordinal = parent.leading.length ∧
  parent.site.stage.inputs = List.range parent.site.stage.program.inputs.length ∧
  Prefix.Valid (List.range parent.site.stage.program.inputs.length) parent.leading ∧
  parent.site.stage.output ∉ List.range parent.site.stage.program.inputs.length ++ parent.leading.map Prefix.Instruction.result ∧
  genericWellFormed parent.site.stage parent.site.shape parent.site.names parent.site.unused ∧
  checkReferences (parent.leading.map Prefix.Instruction.emit) parent.site.stage.program parent.site.names parent.references = true ∧
  checkEligible (parent.references.map (·.slot)) parent.site.stage.program.inputs.length parent.site.stage.program.instructions = true ∧
  (∃ (index : Nat), parent.leading[index]? =
    some (Prefix.Instruction.empty parent.site.initializer parent.site.shape parent.site.stage.program.output)) ∧
  (∀ id ∈ parent.site.names, id ∉ parent.references.map (·.sourceId) →
    id ∉ List.range parent.site.stage.program.inputs.length ++ parent.leading.map Prefix.Instruction.result ++
      [parent.site.stage.output]) ∧
  parent.site.stage.inputTypes = List.replicate parent.site.stage.program.inputs.length (typeTag parent.tensorType) ∧
  parent.site.stage.outputType = typeTag parent.tensorType ∧
  parent.site.unused ∉ List.range parent.site.stage.program.inputs.length ++
    parent.leading.map Prefix.Instruction.result ++ [parent.site.stage.output]

/-- Complete actual parent buffers, not scalar hints, are acceptance inputs. -/
def Parent.Accepts (parent : Parent) (actual retained : Module) : Prop :=
  actual = parent.module ∧ retained = actual ∧ parent.Valid

/-- Candidate source interpretation of the entire admitted parent. Runtime
arguments are initialized tensors; the source prefix computes scalar captures
and never consumes the uninitialized tensor.empty elements. -/
inductive ParentExecution {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) : Module → List (Tensor elements α) → Tensor elements α → Prop where
  | evaluated (parent : Parent) (valid : parent.Valid) (initialCells : Prefix.Cells α)
      (arguments : List (Tensor elements α)) (arity : arguments.length = parent.site.stage.program.inputs.length)
      (extent : elements = parent.site.shape.foldl (· * ·) 1) :
    ParentExecution meaning parent.module arguments
      (sourceTensor meaning parent.site.stage.program parent.site.names parent.references parent.leading initialCells arguments)

/-- Complete-parent source interpretation agrees with the independently checked
closed function's tensor lifting. All operation meanings remain common explicit
parameters. This restricted interpretation grants no unconditional pass claim. -/
theorem checked_parent_agrees_closed {α : Type} [Inhabited α] {elements : Nat}
    {parent : Parent} {actual retained closedSource declaration : Module}
    {sourceName declarationName : String} {sourceVisibility declarationVisibility : Option String}
    (accepted : parent.Accepts actual retained)
    (closed : checkScalar closedSource declaration sourceName declarationName sourceVisibility declarationVisibility
      parent.site.stage.program = true)
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α) (arguments : List (Tensor elements α))
    (arity : arguments.length = parent.site.stage.program.inputs.length)
    (extent : elements = parent.site.shape.foldl (· * ·) 1) :
    actual = retained ∧
    ParentExecution meaning actual arguments (tensorEvaluate parent.site.stage.program meaning arguments) ∧
    (∀ index : Fin elements,
      ScalarExecution meaning closedSource
        (fun slot => (arguments[slot]?.map (· index)).getD default)
        (tensorEvaluate parent.site.stage.program meaning arguments index) ∧
      ScalarExecution meaning declaration
        (fun slot => (arguments[slot]?.map (· index)).getD default)
        (tensorEvaluate parent.site.stage.program meaning arguments index)) := by
  obtain ⟨actual, retained, valid⟩ := accepted
  refine ⟨retained.symm, ?_, ?_⟩
  · rw [actual]
    have equal := sourceTensor_agrees_closed meaning arguments initialCells
      (show Prefix.Matches (parent.leading.map Prefix.Instruction.emit) parent.leading from rfl)
      valid.2.2.1 valid.2.2.2.2.2.1 valid.2.2.2.2.2.2.1 valid.2.2.2.2.1.1
    rw [← equal]
    exact .evaluated parent valid initialCells arguments arity extent
  · intro index
    exact checkScalar_sound closed meaning (fun slot => (arguments[slot]?.map (· index)).getD default)

end Merlin.GraphRegionsDeclared.Capture
