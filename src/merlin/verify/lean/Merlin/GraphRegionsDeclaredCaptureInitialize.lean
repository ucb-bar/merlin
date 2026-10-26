/- Construct the cached
literal domain instead of assuming initialized-success/global typing. The
materialized values are tied to dominating source definitions by CaptureMatch;
this file does not assert a native implementation or frozen Core.Sem bridge. -/
import Merlin.GraphRegionsDeclaredCaptureMatch

namespace Merlin.GraphRegionsDeclared.Capture
open Merlin.GraphRegionsDeclared

def Eligible (captures : List Nat) : Nat → List Instruction → Prop
  | _, [] => True
  | next, instruction :: rest =>
    (next ∈ captures → instruction.operands = [] ∧
      ∃ width literal, instruction.primitive = .constant width literal) ∧
    Eligible captures (next + 1) rest

def checkEligible (captures : List Nat) : Nat → List Instruction → Bool
  | _, [] => true
  | next, instruction :: rest =>
    (if next ∈ captures then decide (instruction.operands = []) && isLiteral instruction.primitive else true) &&
      checkEligible captures (next + 1) rest

theorem isLiteral_sound {primitive : Primitive} (accepted : isLiteral primitive = true) :
    ∃ width literal, primitive = .constant width literal := by
  cases primitive with
  | constant width literal => exact ⟨width, literal, rfl⟩
  | binary _ _ | cast _ _ _ => simp [isLiteral] at accepted

theorem checkEligible_sound {captures : List Nat} {next : Nat} {instructions : List Instruction}
    (accepted : checkEligible captures next instructions = true) : Eligible captures next instructions := by
  induction instructions generalizing next with
  | nil => trivial
  | cons instruction rest ih =>
    simp only [checkEligible, Bool.and_eq_true] at accepted
    refine ⟨?_, ih accepted.2⟩
    intro member
    simp only [member, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at accepted
    exact ⟨accepted.1.1, isLiteral_sound accepted.1.2⟩

/-- Model of source-prefix literal initialization in canonical SSA coordinates.
Only slots whose actual defining literal is checked as an outer capture are
cached. This is a source evaluator, never a substituted machine implementation. -/
def cacheLiterals {α : Type} (meaning : Primitive → List α → α) (captures : List Nat) :
    Nat → List Instruction → Environment α → Environment α
  | _, [], arguments => arguments
  | next, instruction :: rest, arguments =>
    let tail := cacheLiterals meaning captures (next + 1) rest arguments
    if next ∈ captures then bind tail next (meaning instruction.primitive []) else tail

theorem initialize_ready {α : Type} (meaning : Primitive → List α → α) (captures : List Nat)
    {next : Nat} {instructions : List Instruction} (eligible : Eligible captures next instructions)
    (arguments : Environment α) :
    Ready meaning captures next instructions (cacheLiterals meaning captures next instructions arguments) := by
  induction instructions generalizing next with
  | nil => trivial
  | cons instruction rest ih =>
    have tail := ih eligible.2
    by_cases member : next ∈ captures
    · simp only [cacheLiterals, member, ↓reduceIte]
      refine ⟨?_, ready_bind_earlier meaning captures tail (Nat.lt_succ_self _) _⟩
      intro _
      obtain ⟨empty, literal⟩ := eligible.1 member
      exact ⟨empty, literal, by simp [bind]⟩
    · simp only [cacheLiterals, member, ↓reduceIte]
      exact ⟨fun impossible => False.elim (member impossible), tail⟩

theorem initialize_frame {α : Type} (meaning : Primitive → List α → α) (captures : List Nat)
    {next slot : Nat} (instructions : List Instruction) (arguments : Environment α)
    (earlier : slot < next) : cacheLiterals meaning captures next instructions arguments slot = arguments slot := by
  induction instructions generalizing next with
  | nil => rfl
  | cons instruction rest ih =>
    have different : slot ≠ next := Nat.ne_of_lt earlier
    by_cases member : next ∈ captures
    · simp only [cacheLiterals, member, ↓reduceIte, bind, different]
      exact ih (Nat.lt_trans earlier (Nat.lt_succ_self _))
    · simp only [cacheLiterals, member, ↓reduceIte]
      exact ih (Nat.lt_trans earlier (Nat.lt_succ_self _))

/-- Every initialized runtime argument environment has an explicitly constructed
source literal environment. Closure agreement follows from the checked program
and actual cached constant behavior; no existential success premise is used. -/
theorem initialized_closure_result {α : Type} (meaning : Primitive → List α → α) (captures : List Nat)
    (program : Program) (valid : program.Valid)
    (eligible : checkEligible captures program.inputs.length program.instructions = true)
    (arguments : Environment α) :
    sourceRun meaning captures program.inputs.length program.instructions
      (cacheLiterals meaning captures program.inputs.length program.instructions arguments) program.returned =
        program.evaluate meaning arguments := by
  apply closed_result meaning captures program valid _ arguments
  · exact initialize_ready meaning captures (checkEligible_sound eligible) arguments
  · intro index width found
    have bound := (List.getElem?_eq_some_iff.mp found).choose
    exact initialize_frame meaning captures program.instructions arguments bound

theorem ready_lookup {α : Type} {meaning : Primitive → List α → α} {captures : List Nat}
    {next index : Nat} {instructions : List Instruction} {environment : Environment α} {instruction : Instruction}
    (ready : Ready meaning captures next instructions environment)
    (found : instructions[index]? = some instruction) (member : next + index ∈ captures) :
    environment (next + index) = meaning instruction.primitive [] := by
  induction instructions generalizing next index with
  | nil => simp at found
  | cons first rest ih =>
    cases index with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst instruction
      simpa using (ready.1 (by simpa using member)).2.2
    | succ index =>
      simp only [List.getElem?_cons_succ] at found
      have nextMember : (next + 1) + index ∈ captures := by simpa only [Nat.add_assoc, Nat.add_comm 1] using member
      have result := ih ready.2 found nextMember
      simpa only [Nat.add_assoc, Nat.add_comm 1] using result

/-- The cached values are justified by actual checked dominating definitions,
not an externally asserted "constant" flag or an arbitrary source-success fact. -/
theorem initialized_reference_matches_definition {α : Type}
    (meaning : Primitive → List α → α) (arguments : Environment α)
    {leading : List MerlinVerify.Op} {program : Program} {names : List Nat} {references : List Reference}
    (matched : checkReferences leading program names references = true)
    (eligible : checkEligible (references.map (·.slot)) program.inputs.length program.instructions = true)
    {reference : Reference} (member : reference ∈ references) :
    ∃ (width : Nat) (literal : Int) (instruction : Instruction),
      instruction.primitive = .constant width literal ∧
      leading[reference.prefixIndex]? = some (instruction.operation reference.sourceId) ∧
      cacheLiterals meaning (references.map (·.slot)) program.inputs.length program.instructions arguments reference.slot =
        meaning (.constant width literal) [] := by
  obtain ⟨lower, width, literal, instruction, found, _, primitive, _, definition⟩ :=
    checkReference_sound (checkReferences_each matched reference member)
  have ready := initialize_ready meaning (references.map (·.slot)) (checkEligible_sound eligible) arguments
  have slotMember : reference.slot ∈ references.map (·.slot) := List.mem_map.mpr ⟨reference, member, rfl⟩
  have position : program.inputs.length + (reference.slot - program.inputs.length) = reference.slot := by
    rw [Nat.add_comm]
    exact Nat.sub_add_cancel lower
  have cached := ready_lookup ready found (by simpa only [position] using slotMember)
  rw [position, primitive] at cached
  exact ⟨width, literal, instruction, primitive, definition, cached⟩

end Merlin.GraphRegionsDeclared.Capture
