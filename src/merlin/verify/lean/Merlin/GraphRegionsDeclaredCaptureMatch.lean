/- Actual dominating-definition and captured-region syntax checks. This additive
candidate does not change either frozen declared-call cut. Captures are checked
against actual prefix operations; a hint cannot turn a body operation into an
unproved constant or hide a read of the output initializer. -/
import Merlin.GraphRegionsDeclaredCapture
import Merlin.GraphRegionsDeclaredProjection

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegionsDeclared

structure Reference where
  slot : Nat
  sourceId : Nat
  prefixIndex : Nat
  deriving DecidableEq, Repr

def isLiteral : Primitive → Bool
  | .constant .. => true
  | _ => false

def checkReference (leading : List Op) (program : Program) (names : List Nat) (reference : Reference) : Bool :=
  decide (program.inputs.length ≤ reference.slot) &&
    match program.instructions[reference.slot - program.inputs.length]?, names[reference.slot]?, leading[reference.prefixIndex]? with
    | some instruction, some name, some definition =>
      decide (name = reference.sourceId ∧ instruction.operands = []) &&
        isLiteral instruction.primitive && instructionMatches instruction reference.sourceId definition
    | _, _, _ => false

/-- Every accepted capture has an actual earlier constant definition with exact
type/literal/properties and matching SSA identity. These are recomputed facts. -/
theorem checkReference_sound {leading : List Op} {program : Program} {names : List Nat} {reference : Reference}
    (accepted : checkReference leading program names reference = true) :
    program.inputs.length ≤ reference.slot ∧
    ∃ width literal instruction,
      program.instructions[reference.slot - program.inputs.length]? = some instruction ∧
      names[reference.slot]? = some reference.sourceId ∧
      instruction.primitive = .constant width literal ∧ instruction.operands = [] ∧
      leading[reference.prefixIndex]? = some (instruction.operation reference.sourceId) := by
  simp only [checkReference, Bool.and_eq_true, decide_eq_true_eq] at accepted
  refine ⟨accepted.1, ?_⟩
  split at accepted
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at accepted
    rename_i instruction name definition found named defined
    have literal := accepted.2.1.2
    cases primitive : instruction.primitive with
    | constant width value =>
      exact ⟨width, value, instruction, found, by simpa only [accepted.2.1.1.1] using named,
        primitive, accepted.2.1.1.2, by rw [defined, instructionMatches_sound accepted.2.2]⟩
    | binary _ _ | cast _ _ _ => simp [isLiteral, primitive] at literal
  · simp at accepted

def checkReferences (leading : List Op) (program : Program) (names : List Nat) (references : List Reference) : Bool :=
  decide ((references.map (·.slot)).Nodup) && references.all (checkReference leading program names)

theorem checkReferences_each {leading : List Op} {program : Program} {names : List Nat} {references : List Reference}
    (accepted : checkReferences leading program names references = true) :
    ∀ reference ∈ references, checkReference leading program names reference = true := by
  simp only [checkReferences, Bool.and_eq_true, List.all_eq_true] at accepted
  exact accepted.2

def sourceOperations (program : Program) (names captures : List Nat) : Nat → List Instruction → List Op
  | _, [] => [.mk "linalg.yield" [] [renameSlot names program.returned] [] [] [] []]
  | next, instruction :: rest =>
    if next ∈ captures then sourceOperations program names captures (next + 1) rest else
      .mk instruction.primitive.name [⟨renameSlot names next, .int instruction.primitive.width⟩]
        (instruction.operands.map (renameSlot names)) instruction.primitive.attrs [] [] [] ::
        sourceOperations program names captures (next + 1) rest

def originalGeneric (site : GenericSite) (references : List Reference) : Op :=
  let arguments := ((List.range site.stage.program.inputs.length).zip site.stage.program.inputs).map
    (fun (slot, width) => (⟨renameSlot site.names slot, .int width⟩ : Value)) ++
      [⟨site.unused, .int site.stage.program.output⟩]
  match genericOperation site.stage site.shape site.names site.initializer site.unused with
  | .mk name results operands attrs props _ successors =>
    .mk name results operands attrs props [.mk [.mk arguments
      (sourceOperations site.stage.program site.names (references.map (·.slot))
        site.stage.program.inputs.length site.stage.program.instructions)]] successors

def RegionBinding (leading : List Op) (actual : Op) (site : GenericSite) (references : List Reference) : Prop :=
  actual = originalGeneric site references ∧
  genericWellFormed site.stage site.shape site.names site.unused ∧
  checkReferences leading site.stage.program site.names references = true

def ProgramBinding (actual retained : Module) (site : GenericSite) (references : List Reference) : Prop :=
  actual = retained ∧
  match actual.ops with
  | [.mk "func.func" [] [] _ [] [.mk [.mk _ operations]] []] =>
    ∃ operation, operations[site.stage.ordinal]? = some operation ∧
      RegionBinding (operations.take site.stage.ordinal) operation site references
  | _ => False

/-- Complete parent program binding retains every observer and locates the
actual scalar region and dominating prefix. Execution is supplied by the cached
source evaluator, with literal initialization proved separately. -/
theorem program_binding_retained {actual retained : Module} {site : GenericSite} {references : List Reference}
    (accepted : ProgramBinding actual retained site references) : actual = retained := accepted.1

end Merlin.GraphRegionsDeclared.Capture
