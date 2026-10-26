/- Actual complete scalar declaration checks supply the meaning used by each
instruction contract. Native declaration fidelity/commit/frame remain explicit
local premises; no whole-region agreement or numerical law is assumed. -/
import Merlin.GraphRegionsDeclaredCaptureProjection

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

structure DeclarationPair where
  source : Module
  declaration : Module
  sourceName : String
  declarationName : String
  sourceVisibility : Option String
  declarationVisibility : Option String

def DeclarationPair.check (pair : DeclarationPair) (program : Program) : Bool :=
  checkScalar pair.source pair.declaration pair.sourceName pair.declarationName
    pair.sourceVisibility pair.declarationVisibility program

def checkDeclarations (items : List CallStage) (declarations : CallStage → DeclarationPair) : Bool :=
  items.all (fun item => (declarations item).check item.stage.program)

/-- Tensor lifting of execution of the actual declared Module, with initialized
mathematical scalar arguments. It does not identify the native implementation. -/
def DeclarationExecution {α : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α) (declaration : Module)
    (arguments : List (Tensor elements α)) (result : Tensor elements α) : Prop :=
  ∀ index, ScalarExecution primitive declaration
    (fun slot => (arguments[slot]?.map (· index)).getD default) (result index)

theorem checked_declaration_execution {α : Type} [Inhabited α] {elements : Nat}
    {pair : DeclarationPair} {program : Program} (accepted : pair.check program = true)
    (primitive : Primitive → List α → α) (arguments : List (Tensor elements α)) :
    DeclarationExecution primitive pair.declaration arguments (tensorEvaluate program primitive arguments) := by
  intro index
  exact (checkScalar_sound accepted primitive (fun slot => (arguments[slot]?.map (· index)).getD default)).2

/-- Fidelity is to the ACTUAL submitted declaration's execution, not to a
program hint or qualification ID. Completion/no-fault and disjoint frame are
local instruction premises. They remain independently reviewable/unproved. -/
structure DeclaredInstructionContract {α M : Type} [Inhabited α] {elements : Nat}
    (read : M → Window → Option (Tensor elements α)) (invoke : BoundCall → M → Option M)
    (primitive : Primitive → List α → α) (call : BoundCall) (declaration : Module) : Prop where
  capture_declared_commit : ∀ memory arguments result,
    call.reads.map (read memory) = arguments.map some →
    DeclarationExecution primitive declaration arguments result →
    ∃ next, invoke call memory = some next ∧ read next call.output = some result ∧
      ∀ window, window.Disjoint call.output → read next window = read memory window

theorem instruction_contract_from_declaration {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {call : BoundCall} {pair : DeclarationPair} {program : Program}
    (accepted : pair.check program = true)
    (contract : DeclaredInstructionContract read invoke primitive call pair.declaration) :
    InstructionContract read invoke primitive call program := by
  constructor
  intro memory arguments inputs
  exact contract.capture_declared_commit memory arguments (tensorEvaluate program primitive arguments) inputs
    (checked_declaration_execution accepted primitive arguments)

theorem checked_declaration_contracts {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {items : List CallStage} {declarations : CallStage → DeclarationPair}
    (accepted : checkDeclarations items declarations = true)
    (contracts : ∀ item ∈ items,
      DeclaredInstructionContract read invoke primitive item.call (declarations item).declaration) :
    ∀ item ∈ items, InstructionContract read invoke primitive item.call item.stage.program := by
  have all : ∀ item ∈ items, (declarations item).check item.stage.program = true := by
    simpa only [checkDeclarations, List.all_eq_true] using accepted
  intro item member
  exact instruction_contract_from_declaration (all item member) (contracts item member)

end Merlin.GraphRegionsDeclared.Capture
