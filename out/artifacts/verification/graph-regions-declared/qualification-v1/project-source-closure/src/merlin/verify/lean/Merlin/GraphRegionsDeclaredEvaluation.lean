/- Ordered scalar SSA evaluation: only actual typed operand slots are read.
This supplies read-locality for declared pointwise nodes without CSE/DCE or an
arithmetic rewrite. Input equality and initializedness are different obligations. -/
import Merlin.GraphRegionsDeclaredMatch

namespace Merlin.GraphRegionsDeclared

def EnvironmentAgree {α : Type} (widths : List Nat) (first second : Environment α) : Prop :=
  ∀ index width, widths[index]? = some width → first index = second index

theorem operands_agree {α : Type} {widths : List Nat} {instruction : Instruction} {first second : Environment α}
    (agree : EnvironmentAgree widths first second)
    (typed : instruction.operands.map (widths[·]?) = instruction.primitive.inputs.map some) :
    instruction.operands.map first = instruction.operands.map second := by
  apply List.map_congr_left
  intro index member
  cases found : widths[index]? with
  | none =>
    have missing : none ∈ instruction.operands.map (widths[·]?) := List.mem_map.mpr ⟨index, member, found⟩
    rw [typed] at missing
    simp at missing
  | some width => exact agree index width found

theorem environment_agree_append {α : Type} {widths : List Nat} {first second : Environment α}
    (agree : EnvironmentAgree widths first second) (width : Nat) (left right : α) (same : left = right) :
    EnvironmentAgree (widths ++ [width]) (bind first widths.length left) (bind second widths.length right) := by
  intro index type found
  by_cases new : index = widths.length
  · simp [bind, new, same]
  · have bounds := (List.getElem?_eq_some_iff.mp found).choose
    have le : index ≤ widths.length := Nat.le_of_lt_succ (by simpa using bounds)
    have old : index < widths.length := (Nat.lt_or_eq_of_le le).resolve_right new
    rw [List.getElem?_append_left old] at found
    simpa only [bind, new, ↓reduceIte] using agree index type found

theorem run_agrees {α : Type} (meaning : Primitive → List α → α)
    {widths : List Nat} {instructions : List Instruction}
    (valid : instructionsValid widths instructions) {first second : Environment α}
    (agree : EnvironmentAgree widths first second) :
    EnvironmentAgree (finalWidths widths instructions)
      (run meaning widths.length instructions first) (run meaning widths.length instructions second) := by
  induction instructions generalizing widths first second with
  | nil => exact agree
  | cons instruction rest ih =>
    have operands := operands_agree agree valid.2.1
    have step := environment_agree_append agree instruction.primitive.width
      (meaning instruction.primitive (instruction.operands.map first))
      (meaning instruction.primitive (instruction.operands.map second)) (congrArg _ operands)
    have tail := ih valid.2.2 step
    simpa only [finalWidths, run, List.length_append, List.length_singleton] using tail

/-- Every actual operand/result/return slot is covered by checked static widths;
unmentioned environment cells cannot affect a matched function result. -/
theorem Program.locality {α : Type} (program : Program) (valid : program.Valid)
    (meaning : Primitive → List α → α) {first second : Environment α}
    (agree : EnvironmentAgree program.inputs first second) :
    program.evaluate meaning first = program.evaluate meaning second :=
  run_agrees meaning valid.2.2.2.2.1 agree program.returned program.output valid.2.2.2.2.2

def EnvironmentTyped {α : Type} (typing : Nat → α → Prop) (widths : List Nat) (environment : Environment α) : Prop :=
  ∀ index width, widths[index]? = some width → typing width (environment index)

inductive TypedArguments {α : Type} (typing : Nat → α → Prop) : List Nat → List α → Prop where
  | nil : TypedArguments typing [] []
  | cons {width value widths values} : typing width value → TypedArguments typing widths values →
      TypedArguments typing (width :: widths) (value :: values)

/-- Finer common primitive contract. It states typed total scalar behavior;
this is not an axiom or a native whole-region implementation premise. -/
structure PrimitiveInterpretation (α : Type) (typing : Nat → α → Prop) where
  eval : Primitive → List α → α
  preserves : ∀ primitive, primitive.Valid → ∀ arguments,
    TypedArguments typing primitive.inputs arguments → typing primitive.width (eval primitive arguments)

theorem operands_typed {α : Type} {typing : Nat → α → Prop}
    {widths : List Nat} {instruction : Instruction} {environment : Environment α}
    (typed : EnvironmentTyped typing widths environment)
    (arguments : instruction.operands.map (widths[·]?) = instruction.primitive.inputs.map some) :
    TypedArguments typing instruction.primitive.inputs (instruction.operands.map environment) := by
  generalize instruction.operands = operands at arguments ⊢
  generalize instruction.primitive.inputs = inputs at arguments ⊢
  induction operands generalizing inputs with
  | nil =>
    cases inputs with
    | nil => exact .nil
    | cons _ _ => simp at arguments
  | cons index operands ih =>
    cases inputs with
    | nil => simp at arguments
    | cons width inputs =>
      simp only [List.map_cons, List.cons.injEq] at arguments
      exact .cons (typed index width arguments.1) (ih inputs arguments.2)

theorem environment_typed_append {α : Type} {typing : Nat → α → Prop}
    {widths : List Nat} {environment : Environment α}
    (typed : EnvironmentTyped typing widths environment) (width : Nat) (value : α) (result : typing width value) :
    EnvironmentTyped typing (widths ++ [width]) (bind environment widths.length value) := by
  intro index type found
  by_cases new : index = widths.length
  · subst index
    have same : type = width := by
      simpa only [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero,
        Option.some.injEq] using found.symm
    subst type
    simpa [bind] using result
  · have bounds := (List.getElem?_eq_some_iff.mp found).choose
    have old : index < widths.length := (Nat.lt_or_eq_of_le (Nat.le_of_lt_succ (by simpa using bounds))).resolve_right new
    rw [List.getElem?_append_left old] at found
    simpa [bind, new] using typed index type found

theorem run_typed {α : Type} {typing : Nat → α → Prop} (meaning : PrimitiveInterpretation α typing)
    {widths : List Nat} {instructions : List Instruction} (valid : instructionsValid widths instructions)
    {environment : Environment α} (typed : EnvironmentTyped typing widths environment) :
    EnvironmentTyped typing (finalWidths widths instructions)
      (run meaning.eval widths.length instructions environment) := by
  induction instructions generalizing widths environment with
  | nil => exact typed
  | cons instruction rest ih =>
    have result := meaning.preserves instruction.primitive valid.1 _ (operands_typed typed valid.2.1)
    have step := environment_typed_append typed instruction.primitive.width _ result
    simpa only [finalWidths, run, List.length_append, List.length_singleton] using ih valid.2.2 step

theorem Program.typed {α : Type} {typing : Nat → α → Prop} (program : Program) (valid : program.Valid)
    (meaning : PrimitiveInterpretation α typing) {environment : Environment α}
    (typed : EnvironmentTyped typing program.inputs environment) :
    typing program.output (program.evaluate meaning.eval environment) :=
  run_typed meaning valid.2.2.2.2.1 typed program.returned program.output valid.2.2.2.2.2

end Merlin.GraphRegionsDeclared
