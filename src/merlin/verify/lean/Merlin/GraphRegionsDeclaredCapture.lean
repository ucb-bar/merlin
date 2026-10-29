/- Additive captured-literal closure reasoning. This is a new candidate, separate
from both frozen declared-call cuts. A source environment caches outer literals;
the closed scalar function materializes them at their captured SSA slots.
Pure, total common primitive interpretation is explicit. No arithmetic law,
native implementation, frozen Core.Sem identification or source-success axiom is
assumed. Actual parent-definition matching belongs to the following checker. -/
import Merlin.GraphRegionsDeclaredEvaluation

namespace Merlin.GraphRegionsDeclared.Capture
open Merlin.GraphRegionsDeclared

/-- `true` means this canonical slot denotes an outer source literal rather
than a body-local instruction result. This hint must later be checked against
the actual dominating constant definition and complete source-region syntax. -/
def Ready {α : Type} (meaning : Primitive → List α → α) (captured : List Nat) :
    Nat → List Instruction → Environment α → Prop
  | _, [], _ => True
  | next, instruction :: rest, environment =>
    (next ∈ captured → instruction.operands = [] ∧
      (∃ width literal, instruction.primitive = .constant width literal) ∧
      environment next = meaning instruction.primitive []) ∧
    Ready meaning captured (next + 1) rest environment

/-- Original canonical source evaluation reads an already defined outer literal;
it does not execute a second source constant operation inside the generic. All
body-local operations still execute in their original order. -/
def sourceRun {α : Type} (meaning : Primitive → List α → α) (captured : List Nat) :
    Nat → List Instruction → Environment α → Environment α
  | _, [], environment => environment
  | next, instruction :: rest, environment =>
    if next ∈ captured then sourceRun meaning captured (next + 1) rest environment
    else sourceRun meaning captured (next + 1) rest
      (bind environment next (meaning instruction.primitive (instruction.operands.map environment)))

theorem bind_unchanged {α : Type} (environment : Environment α) (slot : Nat) (value : α)
    (same : environment slot = value) : bind environment slot value = environment := by
  funext key
  by_cases equal : key = slot
  · subst key
    simp [bind, same]
  · simp [bind, equal]

/-- Earlier fresh SSA writes cannot overwrite a future cached capture slot. -/
theorem ready_bind_earlier {α : Type} (meaning : Primitive → List α → α) (captured : List Nat)
    {next slot : Nat} {instructions : List Instruction} {environment : Environment α}
    (ready : Ready meaning captured next instructions environment) (earlier : slot < next) (value : α) :
    Ready meaning captured next instructions (bind environment slot value) := by
  induction instructions generalizing next with
  | nil => trivial
  | cons instruction rest ih =>
    refine ⟨?_, ih ready.2 (Nat.lt_trans earlier (Nat.lt_succ_self _))⟩
    intro member
    obtain ⟨operands, literal, cached⟩ := ready.1 member
    refine ⟨operands, literal, ?_⟩
    have different : next ≠ slot := Ne.symm (Nat.ne_of_lt earlier)
    simpa [bind, different] using cached

/-- Inserting a correctly matched pure literal at its closure slot changes no
value: that slot already contains the exact value of its actual outer definition.
The conclusion is full environment equality, not just one output sample. -/
theorem sourceRun_eq_run {α : Type} (meaning : Primitive → List α → α) (captured : List Nat)
    {next : Nat} {instructions : List Instruction} {environment : Environment α}
    (ready : Ready meaning captured next instructions environment) :
    sourceRun meaning captured next instructions environment = run meaning next instructions environment := by
  induction instructions generalizing next environment with
  | nil => rfl
  | cons instruction rest ih =>
    by_cases external : next ∈ captured
    · obtain ⟨operands, _, cached⟩ := ready.1 external
      have noChange : bind environment next
          (meaning instruction.primitive (instruction.operands.map environment)) = environment := by
        rw [operands]
        exact bind_unchanged environment next _ cached
      simp only [sourceRun, external, ↓reduceIte, run, noChange]
      exact ih ready.2
    · simp only [sourceRun, external, ↓reduceIte, run]
      exact ih (ready_bind_earlier meaning captured ready.2 (Nat.lt_succ_self _) _)

/-- Runtime formal arguments need not contain the cached future SSA slots. The
existing checked scalar locality theorem discharges that difference. -/
theorem closed_result {α : Type} (meaning : Primitive → List α → α) (captured : List Nat)
    (program : Program) (valid : program.Valid) (source arguments : Environment α)
    (ready : Ready meaning captured program.inputs.length program.instructions source)
    (sameInputs : EnvironmentAgree program.inputs source arguments) :
    sourceRun meaning captured program.inputs.length program.instructions source program.returned =
      program.evaluate meaning arguments := by
  rw [sourceRun_eq_run meaning captured ready]
  exact program.locality valid meaning sameInputs

end Merlin.GraphRegionsDeclared.Capture
