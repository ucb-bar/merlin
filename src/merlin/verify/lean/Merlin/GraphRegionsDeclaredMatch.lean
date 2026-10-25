/- Complete closed scalar AST matching. Names/visibility are explicit header
parameters; operation bodies, properties, types and ordered SSA are exact.
No digest appears in this checker or its soundness statement. -/
import Merlin.GraphRegionsDeclaredScalar

namespace Merlin.GraphRegionsDeclared
open MerlinVerify

def instructionsValid : List Nat → List Instruction → Prop
  | _, [] => True
  | widths, instruction :: rest => instruction.primitive.Valid ∧
      instruction.operands.map (widths[·]?) = instruction.primitive.inputs.map some ∧
      instructionsValid (widths ++ [instruction.primitive.width]) rest

def finalWidths : List Nat → List Instruction → List Nat
  | widths, [] => widths
  | widths, instruction :: rest => finalWidths (widths ++ [instruction.primitive.width]) rest

def Program.Valid (program : Program) : Prop :=
  (∀ width ∈ program.inputs, 0 < width ∧ width ≤ 256 ∧ width % 8 = 0) ∧
    0 < program.output ∧ program.output ≤ 256 ∧ program.output % 8 = 0 ∧
    instructionsValid program.inputs program.instructions ∧
    (finalWidths program.inputs program.instructions)[program.returned]? = some program.output

def returnOp (returned : Nat) : Op := .mk "func.return" [] [returned] [] [] [] []

def emitOps (returned : Nat) : Nat → List Instruction → List Op
  | _, [] => [returnOp returned]
  | next, instruction :: rest => instruction.operation next :: emitOps returned (next + 1) rest

def Program.arguments (program : Program) : List Value :=
  ((List.range program.inputs.length).zip program.inputs).map fun (id, width) => ⟨id, .int width⟩

def Program.body (program : Program) : Region :=
  .mk [.mk program.arguments (emitOps program.returned program.inputs.length program.instructions)]

def Program.attrs (program : Program) (name : String) (visibility : Option String) : List (String × Attr) :=
  [("function_type", .type (.function (program.inputs.map Ty.int) [.int program.output])),
   ("sym_name", .str name)] ++ visibility.toList.map fun value => ("sym_visibility", .str value)

def Program.module (program : Program) (name : String) (visibility : Option String) : Module :=
  ⟨[.mk "func.func" [] [] (program.attrs name visibility) [] [program.body] []], [], []⟩

def instructionMatches (instruction : Instruction) (next : Nat) : Op → Bool
  | .mk name [⟨id, type⟩] operands attrs props [] [] =>
    decide (name = instruction.primitive.name ∧ id = next ∧ type = .int instruction.primitive.width ∧
      operands = instruction.operands ∧ attrs = instruction.primitive.attrs ∧ props = [])
  | _ => false

theorem instructionMatches_sound {instruction : Instruction} {next : Nat} {actual : Op}
    (accepted : instructionMatches instruction next actual = true) : actual = instruction.operation next := by
  unfold instructionMatches at accepted
  split at accepted
  · simp only [decide_eq_true_eq] at accepted
    obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := accepted
    rfl
  · contradiction

def terminalMatches (returned : Nat) : Op → Bool
  | .mk name [] operands [] [] [] [] => decide (name = "func.return" ∧ operands = [returned])
  | _ => false

theorem terminalMatches_sound {returned : Nat} {actual : Op}
    (accepted : terminalMatches returned actual = true) : actual = returnOp returned := by
  unfold terminalMatches at accepted
  split at accepted
  · simp only [decide_eq_true_eq] at accepted
    obtain ⟨rfl, rfl⟩ := accepted
    rfl
  · contradiction

def operationsMatch (returned : Nat) : Nat → List Instruction → List Op → Bool
  | _, [], [actual] => terminalMatches returned actual
  | next, instruction :: rest, actual :: tail =>
    instructionMatches instruction next actual && operationsMatch returned (next + 1) rest tail
  | _, _, _ => false

theorem operationsMatch_sound {returned next : Nat} {instructions : List Instruction} {actual : List Op}
    (accepted : operationsMatch returned next instructions actual = true) :
    actual = emitOps returned next instructions := by
  induction instructions generalizing next actual with
  | nil =>
    cases actual with
    | nil => simp [operationsMatch] at accepted
    | cons op tail =>
      cases tail with
      | nil => rw [terminalMatches_sound accepted]; rfl
      | cons _ _ => simp [operationsMatch] at accepted
  | cons instruction rest ih =>
    cases actual with
    | nil => simp [operationsMatch] at accepted
    | cons op tail =>
      simp only [operationsMatch, Bool.and_eq_true] at accepted
      rw [instructionMatches_sound accepted.1, ih accepted.2]
      rfl

private def valuePairs (values : List Value) : List (Nat × Ty) := values.map fun value => (value.id, value.ty)

private theorem valuePairs_injective {first second : List Value} (same : valuePairs first = valuePairs second) :
    first = second := by
  have rebuilt := congrArg (List.map fun (id, type) => (⟨id, type⟩ : Value)) same
  simp only [valuePairs, List.map_map, Function.comp_def] at rebuilt
  change List.map id first = List.map id second at rebuilt
  simpa using rebuilt

def bodyMatches (program : Program) : Region → Bool
  | .mk [.mk arguments operations] =>
    decide (valuePairs arguments = valuePairs program.arguments) &&
      operationsMatch program.returned program.inputs.length program.instructions operations
  | _ => false

theorem bodyMatches_sound {program : Program} {actual : Region}
    (accepted : bodyMatches program actual = true) : actual = program.body := by
  unfold bodyMatches at accepted
  split at accepted
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at accepted
    rw [valuePairs_injective accepted.1, operationsMatch_sound accepted.2]
    rfl
  · contradiction

def Program.matches (program : Program) (name : String) (visibility : Option String) (actual : Module) : Bool :=
  decide (actual.attrs = [] ∧ actual.props = []) &&
    match actual.ops with
    | [.mk actualName [] [] attrs [] [region] []] =>
      decide (actualName = "func.func" ∧ attrs = program.attrs name visibility) && bodyMatches program region
    | _ => false

theorem Program.matches_sound {program : Program} {name : String} {visibility : Option String} {actual : Module}
    (accepted : program.matches name visibility actual = true) : actual = program.module name visibility := by
  simp only [Program.matches, Bool.and_eq_true, decide_eq_true_eq] at accepted
  obtain ⟨attributes, matched⟩ := accepted
  split at matched
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at matched
    obtain ⟨⟨rfl, attrs⟩, body⟩ := matched
    have bodyEqual := bodyMatches_sound body
    cases actual
    simp_all [Program.module]
  · simp at matched

theorem operationsMatch_refl (returned next : Nat) (instructions : List Instruction) :
    operationsMatch returned next instructions (emitOps returned next instructions) = true := by
  induction instructions generalizing next with
  | nil => simp [emitOps, operationsMatch, terminalMatches, returnOp]
  | cons instruction rest ih =>
    simp [emitOps, operationsMatch, instructionMatches, Instruction.operation, ih]

theorem Program.matches_refl (program : Program) (name : String) (visibility : Option String) :
    program.matches name visibility (program.module name visibility) = true := by
  simp [Program.matches, Program.module, bodyMatches, Program.body, operationsMatch_refl]

instance (primitive : Primitive) : Decidable primitive.Valid := by
  cases primitive with
  | constant _ _ => unfold Primitive.Valid; infer_instance
  | binary _ _ => unfold Primitive.Valid; infer_instance
  | cast _ _ kind => cases kind <;> unfold Primitive.Valid <;> infer_instance

def instructionsValidDec : (widths : List Nat) → (instructions : List Instruction) →
    Decidable (instructionsValid widths instructions)
  | _, [] => isTrue trivial
  | widths, instruction :: rest =>
    letI := instructionsValidDec (widths ++ [instruction.primitive.width]) rest
    inferInstanceAs (Decidable (instruction.primitive.Valid ∧
      instruction.operands.map (widths[·]?) = instruction.primitive.inputs.map some ∧ instructionsValid _ rest))

instance (widths : List Nat) (instructions : List Instruction) : Decidable (instructionsValid widths instructions) :=
  instructionsValidDec widths instructions

instance (program : Program) : Decidable program.Valid := by unfold Program.Valid; infer_instance

def checkScalar (source declaration : Module) (sourceName declarationName : String)
    (sourceVisibility declarationVisibility : Option String) (program : Program) : Bool :=
  program.matches sourceName sourceVisibility source &&
    program.matches declarationName declarationVisibility declaration && decide program.Valid

/-- Proof-producing concrete queries can establish syntactic equality by kernel
conversion without attempting to execute frozen well-founded AST equality. -/
theorem checkScalar_iff {source declaration : Module} {sourceName declarationName : String}
    {sourceVisibility declarationVisibility : Option String} {program : Program} :
    checkScalar source declaration sourceName declarationName sourceVisibility declarationVisibility program = true ↔
      source = program.module sourceName sourceVisibility ∧
      declaration = program.module declarationName declarationVisibility ∧ program.Valid := by
  constructor
  · intro accepted
    simp only [checkScalar, Bool.and_eq_true, decide_eq_true_eq] at accepted
    exact ⟨Program.matches_sound accepted.1.1, Program.matches_sound accepted.1.2, accepted.2⟩
  · rintro ⟨rfl, rfl, valid⟩
    simp only [checkScalar, Program.matches_refl, Bool.true_and, decide_eq_true_eq]
    exact valid

/-- Scalar interpretation of the admitted closed typed DAG under a common
primitive interpretation. This does not define a native machine implementation. -/
inductive ScalarExecution {α : Type} (meaning : Primitive → List α → α) : Module → Environment α → α → Prop where
  | closed (program : Program) (valid : program.Valid) (name : String) (visibility : Option String)
      (arguments : Environment α) :
      ScalarExecution meaning (program.module name visibility) arguments (program.evaluate meaning arguments)

/-- Complete actual AST acceptance supplies one common ordered execution for
every primitive interpretation; hashes and provider qualification IDs are absent. -/
theorem checkScalar_sound {α : Type} {source declaration : Module} {sourceName declarationName : String}
    {sourceVisibility declarationVisibility : Option String} {program : Program}
    (accepted : checkScalar source declaration sourceName declarationName sourceVisibility declarationVisibility program = true)
    (meaning : Primitive → List α → α) (arguments : Environment α) :
    ScalarExecution meaning source arguments (program.evaluate meaning arguments) ∧
      ScalarExecution meaning declaration arguments (program.evaluate meaning arguments) := by
  simp only [checkScalar, Bool.and_eq_true, decide_eq_true_eq] at accepted
  rw [Program.matches_sound accepted.1.1, Program.matches_sound accepted.1.2]
  exact ⟨.closed program accepted.2 sourceName sourceVisibility arguments,
    .closed program accepted.2 declarationName declarationVisibility arguments⟩

end Merlin.GraphRegionsDeclared
