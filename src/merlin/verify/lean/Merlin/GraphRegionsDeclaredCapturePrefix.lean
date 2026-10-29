/- Restricted captured-source prefix execution and initialization.
Restricted actual-prefix scalar projection: integer literal definitions and
static tensor.empty only. Tensor results project to none; this is not a claim
that their elements are initialized. Unknown operations have no constructor.
No source-success/global-typing premise is used to establish captured literals. -/
import Merlin.GraphRegionsDeclaredCaptureContract

namespace Merlin.GraphRegionsDeclared.Capture.Prefix
open MerlinVerify Merlin.GraphRegionsDeclared

inductive Instruction where
  | literal (result width : Nat) (value : Int)
  | empty (result : Nat) (shape : List Nat) (width : Nat)
  deriving DecidableEq, Repr

def Instruction.result : Instruction → Nat
  | .literal result .. | .empty result .. => result

def Instruction.emit : Instruction → Op
  | .literal result width value =>
    .mk "arith.constant" [⟨result, .int width⟩] [] [("value", .int value (.int width))] [] [] []
  | .empty result shape width =>
    .mk "tensor.empty" [⟨result, .tensor (shape.map some) (.int width) none⟩] [] [] [] [] []

def Instruction.Valid : Instruction → Prop
  | .literal _ width _ => 0 < width ∧ width ≤ 256
  | .empty _ shape width => 0 < width ∧ width ≤ 256 ∧ width % 8 = 0 ∧ ∀ dimension ∈ shape, 0 < dimension

def Valid : List Nat → List Instruction → Prop
  | _, [] => True
  | available, instruction :: rest => instruction.Valid ∧ instruction.result ∉ available ∧
    Valid (available ++ [instruction.result]) rest

instance (instruction : Instruction) : Decidable instruction.Valid := by
  cases instruction <;> unfold Instruction.Valid <;> infer_instance

def validDec (available : List Nat) : (program : List Instruction) → Decidable (Valid available program)
  | [] => isTrue trivial
  | instruction :: rest =>
    letI := validDec (available ++ [instruction.result]) rest
    inferInstanceAs (Decidable (instruction.Valid ∧ instruction.result ∉ available ∧
      Valid (available ++ [instruction.result]) rest))

instance (available : List Nat) (program : List Instruction) : Decidable (Valid available program) :=
  validDec available program

/-- Complete actual prefix operations, types and properties must match. This is
a proof-producing structural acceptance relation, not a Python assertion. -/
def Matches (actual : List Op) (program : List Instruction) : Prop := actual = program.map Instruction.emit

abbrev Cells (α : Type) := Nat → Option α

def Instruction.value {α : Type} (meaning : Primitive → List α → α) : Instruction → Option α
  | .literal _ width value => some (meaning (.constant width value) [])
  | .empty .. => none

def run {α : Type} (meaning : Primitive → List α → α) : List Instruction → Cells α → Cells α
  | [], initial => initial
  | instruction :: rest, initial =>
    run meaning rest (bind initial instruction.result (instruction.value meaning))

/-- Source SSA freshness preserves every pre-existing scalar projection, even
when the prefix contains tensor.empty results with uninitialized elements. -/
theorem preserves_available {α : Type} (meaning : Primitive → List α → α)
    {available : List Nat} {program : List Instruction} (valid : Valid available program)
    (initial : Cells α) {slot : Nat} (known : slot ∈ available) :
    run meaning program initial slot = initial slot := by
  induction program generalizing available initial with
  | nil => rfl
  | cons instruction rest ih =>
    have retained : slot ∈ available ++ [instruction.result] := List.mem_append_left _ known
    have different : slot ≠ instruction.result := by
      intro equal
      exact valid.2.1 (equal ▸ known)
    change run meaning rest (bind initial instruction.result (instruction.value meaning)) slot = _
    rw [ih valid.2.2 _ retained]
    simp [bind, different]

/-- A checked dominating integer literal is initialized after execution of the
restricted prefix, for EVERY initial scalar projection. This supplies definedness
constructively; the tensor.empty values need never be read. -/
theorem literal_initialized {α : Type} (meaning : Primitive → List α → α)
    {available : List Nat} {program : List Instruction} {index result width : Nat} {value : Int}
    (valid : Valid available program) (definition : program[index]? = some (.literal result width value))
    (initial : Cells α) : run meaning program initial result = some (meaning (.constant width value) []) := by
  induction program generalizing available index initial with
  | nil => simp at definition
  | cons first rest ih =>
    cases index with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at definition
      subst first
      change run meaning rest (bind initial result (some (meaning (.constant width value) []))) result = _
      rw [preserves_available meaning valid.2.2 _ (List.mem_append_right _ (by simp [Instruction.result]))]
      simp [bind]
    | succ index =>
      simp only [List.getElem?_cons_succ] at definition
      exact ih valid.2.2 definition (bind initial first.result (first.value meaning))

/-- Captured literal initialization is independent of arbitrary runtime tensor
arguments. These remain a separate initialized-input domain at the region. -/
theorem checked_prefix_literal {α : Type} (meaning : Primitive → List α → α)
    {actual : List Op} {program : List Instruction} {available : List Nat}
    (matched : Matches actual program) (valid : Valid available program)
    {index result width : Nat} {value : Int}
    (definition : program[index]? = some (.literal result width value)) (initial : Cells α) :
    actual[index]? = some ((Instruction.literal result width value).emit) ∧
      run meaning program initial result = some (meaning (.constant width value) []) := by
  refine ⟨?_, literal_initialized meaning valid definition initial⟩
  rw [matched, List.getElem?_map, definition]
  rfl

/-- Structural injectivity at a literal definition. It is independent of any
hash, interpretation or value sample. Unknown/tensor operations cannot qualify. -/
theorem emit_eq_literal {instruction : Instruction} {result width : Nat} {value : Int}
    (same : instruction.emit = (Instruction.literal result width value).emit) :
    instruction = .literal result width value := by
  cases instruction with
  | literal id bits literal =>
    have results := congrArg Op.results same
    simp only [Instruction.emit, Op.results] at results
    injection results with field _
    injection field with ids types
    subst id
    injection types with widths
    subst bits
    have attrs := congrArg Op.attrs same
    simp only [Instruction.emit, Op.attrs] at attrs
    injection attrs with entry _
    injection entry with _ payload
    injection payload with literals _
    subst literal
    rfl
  | empty id shape bits =>
    have names := congrArg Op.name same
    have different : "tensor.empty" ≠ "arith.constant" := by decide
    exact False.elim (different (by simpa only [Instruction.emit, Op.name] using names))

theorem literal_lookup_of_match {actual : List Op} {program : List Instruction}
    (matched : Matches actual program) {index result width : Nat} {value : Int}
    (definition : actual[index]? = some ((Instruction.literal result width value).emit)) :
    program[index]? = some (.literal result width value) := by
  rw [matched, List.getElem?_map] at definition
  cases found : program[index]? with
  | none => simp [found] at definition
  | some instruction =>
    rw [found] at definition
    have same : instruction.emit = (Instruction.literal result width value).emit := Option.some.inj definition
    exact congrArg some (emit_eq_literal same)

end Merlin.GraphRegionsDeclared.Capture.Prefix
