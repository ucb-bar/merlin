/- Factor-wide bulk bodies in the SAME partial/effectful SCF semantics.
The log records ORIGINAL body occurrences, not just bulk-loop IVs. -/
import Merlin.GraphLoopsUnroll

set_option autoImplicit false

namespace Merlin.GraphLoopsUnroll

open GraphLoopsSegment

theorem CountCorrect.endpoint {lower upper step count : Nat} (correct : CountCorrect lower upper step count) :
    upper ≤ lower + step * count := by
  cases count with
  | zero => simpa only [CountCorrect, Nat.mul_zero, Nat.add_zero] using correct
  | succ count => exact correct.2

theorem CountCorrect.aligned (lower step count : Nat) (positive : 0 < step) :
    CountCorrect lower (lower + step * count) step count := by
  cases count with
  | zero => simp [CountCorrect]
  | succ count =>
    simp only [CountCorrect, Nat.mul_add, Nat.mul_one]
    omega

theorem CountCorrect.drop {lower upper step first rest : Nat}
    (correct : CountCorrect lower upper step (first + rest)) :
    CountCorrect (lower + step * first) upper step rest := by
  cases rest with
  | zero => simpa only [CountCorrect, Nat.add_zero] using correct.endpoint
  | succ rest =>
    have source : CountCorrect lower upper step ((first + rest) + 1) := by
      simpa only [Nat.add_assoc] using correct
    simpa only [CountCorrect, Nat.mul_add, Nat.add_assoc] using source

abbrev Logged (σ : Type) := σ × List Nat

def GroupBody {σ : Type} (body : Nat → σ → σ → Prop) (step factor index : Nat)
    (before after : Logged σ) : Prop :=
  ∃ trace, Repeat body step factor index before.1 trace after.1 ∧ after.2 = before.2 ++ trace

/-- Exact grouping, including each group's entry/output state and original IVs.
No regrouping law about the arithmetic inside a body is needed. -/
theorem grouped_repeat_iff {σ : Type} (body : Nat → σ → σ → Prop) (step factor groups : Nat) :
    ∀ lower initial final startLog endLog,
      (∃ bases, Repeat (GroupBody body step factor) (step * factor) groups lower
        (initial, startLog) bases (final, endLog)) ↔
      ∃ trace, Repeat body step (groups * factor) lower initial trace final ∧ endLog = startLog ++ trace := by
  induction groups with
  | zero =>
    intro lower initial final startLog endLog
    constructor
    · rintro ⟨bases, run⟩
      cases run
      exact ⟨[], by simpa only [Nat.zero_mul] using Repeat.nil (body := body) (step := step) lower initial, by simp⟩
    · rintro ⟨trace, run, equal⟩
      have zero : Repeat body step 0 lower initial trace final := by simpa only [Nat.zero_mul] using run
      cases zero
      simp only [List.append_nil] at equal
      subst endLog
      exact ⟨[], .nil _ _⟩
  | succ groups induction =>
    intro lower initial final startLog endLog
    constructor
    · rintro ⟨bases, run⟩
      cases run with
      | @cons _ _ _ next _ remainingBases event remaining =>
        rcases next with ⟨middle, middleLog⟩
        obtain ⟨first, firstRun, firstLog⟩ := event
        dsimp only at firstRun firstLog
        obtain ⟨rest, restRun, restLog⟩ :=
          (induction (lower + step * factor) middle final middleLog endLog).mp ⟨remainingBases, remaining⟩
        refine ⟨first ++ rest, ?_, ?_⟩
        · have joined := firstRun.append (groups * factor) final rest restRun
          simpa only [Nat.succ_mul, Nat.add_comm] using joined
        · rw [restLog, firstLog, List.append_assoc]
    · rintro ⟨trace, run, endLogEq⟩
      have normalized : Repeat body step (factor + groups * factor) lower initial trace final := by
        simpa only [Nat.succ_mul, Nat.add_comm] using run
      obtain ⟨middle, first, rest, equal, firstRun, restRun⟩ :=
        Repeat.split body step factor (groups * factor) lower initial final trace normalized
      have finalLog : endLog = (startLog ++ first) ++ rest := by
        rw [endLogEq, equal, List.append_assoc]
      obtain ⟨bases, remaining⟩ :=
        (induction (lower + step * factor) middle final (startLog ++ first) endLog).mpr ⟨rest, restRun, finalLog⟩
      exact ⟨lower :: bases, .cons ⟨first, firstRun, rfl⟩ remaining⟩

/-- The physical bulk SCF loop is connected to the complete original-body log. -/
theorem bulk_iff_repeat {σ : Type} (body : Nat → σ → σ → Prop) (lower step factor groups : Nat)
    (positiveStep : 0 < step) (positiveFactor : 0 < factor) (initial final : σ) (startLog endLog : List Nat) :
    (∃ bases, Execution (GroupBody body step factor) (lower + (step * factor) * groups) (step * factor) lower
      (initial, startLog) bases (final, endLog)) ↔
    ∃ trace, Repeat body step (groups * factor) lower initial trace final ∧ endLog = startLog ++ trace := by
  have count := CountCorrect.aligned lower (step * factor) groups (Nat.mul_pos positiveStep positiveFactor)
  have matching := repeat_iff_execution (GroupBody body step factor) (lower + (step * factor) * groups)
    (step * factor) groups lower (initial, startLog) (final, endLog)
  constructor
  · rintro ⟨bases, executed⟩
    exact (grouped_repeat_iff body step factor groups lower initial final startLog endLog).mp
      ⟨bases, (matching bases count).mpr executed⟩
  · intro repeated
    obtain ⟨bases, repeated⟩ := (grouped_repeat_iff body step factor groups lower initial final startLog endLog).mpr repeated
    exact ⟨bases, (matching bases count).mp repeated⟩

def Tail {σ : Type} (body : Nat → σ → σ → Prop) (lower upper step remainder : Nat)
    (initial : σ) (trace : List Nat) (final : σ) : Prop :=
  if remainder = 0 then trace = [] ∧ final = initial else Execution body upper step lower initial trace final

theorem tail_iff_execution {σ : Type} (body : Nat → σ → σ → Prop) (lower upper step remainder : Nat)
    (correct : CountCorrect lower upper step remainder) (initial final : σ) (trace : List Nat) :
    Tail body lower upper step remainder initial trace final ↔ Execution body upper step lower initial trace final := by
  cases remainder with
  | zero =>
    simp only [Tail, ↓reduceIte]
    constructor
    · rintro ⟨rfl, rfl⟩; exact .done _ correct
    · intro run; exact run.stopped correct
  | succ remainder => simp only [Tail, Nat.succ_ne_zero, ↓reduceIte]

def Emitted {σ : Type} (body : Nat → σ → σ → Prop) (lower upper step count factor groups remainder : Nat)
    (initial : σ) (trace : List Nat) (final : σ) : Prop :=
  if count ≤ factor then Repeat body step count lower initial trace final else
    ∃ middle bulkTrace tailTrace bases,
      Execution (GroupBody body step factor) (lower + (step * factor) * groups) (step * factor) lower
        (initial, []) bases (middle, bulkTrace) ∧
      Tail body (lower + (step * factor) * groups) upper step remainder middle tailTrace final ∧ trace = bulkTrace ++ tailTrace

/-- Both actual production forms preserve complete ordered original-body traces
and the full state. The empty inline form returns the initial state unchanged. -/
theorem emitted_iff_source {σ : Type} (body : Nat → σ → σ → Prop)
    (lower upper step count factor groups remainder : Nat) (positiveStep : 0 < step) (positiveFactor : 0 < factor)
    (counted : CountCorrect lower upper step count) (partition : groups * factor + remainder = count)
    (initial final : σ) (trace : List Nat) :
    Emitted body lower upper step count factor groups remainder initial trace final ↔
      Execution body upper step lower initial trace final := by
  unfold Emitted
  split
  · exact repeat_iff_execution body upper step count lower initial final trace counted
  · have tailCount : CountCorrect (lower + (step * factor) * groups) upper step remainder := by
      have dropped := CountCorrect.drop (show CountCorrect lower upper step (groups * factor + remainder) from partition ▸ counted)
      simpa only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using dropped
    constructor
    · rintro ⟨middle, bulkTrace, tailTrace, bases, bulkRun, tailRun, equal⟩
      have tailRun := (tail_iff_execution body _ upper step remainder tailCount middle final tailTrace).mp tailRun
      obtain ⟨first, firstRun, logEq⟩ :=
        (bulk_iff_repeat body lower step factor groups positiveStep positiveFactor initial middle [] bulkTrace).mp ⟨bases, bulkRun⟩
      simp only [List.nil_append] at logEq
      subst first
      have tailRepeat := (repeat_iff_execution body upper step remainder _ middle final tailTrace tailCount).mpr tailRun
      have adjusted : Repeat body step remainder (lower + step * (groups * factor)) middle tailTrace final := by
        simpa only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using tailRepeat
      have joined := firstRun.append remainder final tailTrace adjusted
      apply (repeat_iff_execution body upper step count lower initial final trace counted).mp
      simpa only [partition, ← equal] using joined
    · intro executed
      have repeated := (repeat_iff_execution body upper step count lower initial final trace counted).mpr executed
      have normalized : Repeat body step (groups * factor + remainder) lower initial trace final := partition ▸ repeated
      obtain ⟨middle, first, rest, equal, firstRun, tailRun⟩ :=
        Repeat.split body step (groups * factor) remainder lower initial final trace normalized
      obtain ⟨bases, bulkRun⟩ :=
        (bulk_iff_repeat body lower step factor groups positiveStep positiveFactor initial middle [] first).mpr
          ⟨first, firstRun, by simp⟩
      refine ⟨middle, first, rest, bases, bulkRun, ?_, equal⟩
      apply (tail_iff_execution body _ upper step remainder tailCount middle final rest).mpr
      apply (repeat_iff_execution body upper step remainder _ middle final rest tailCount).mp
      simpa only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using tailRun

end Merlin.GraphLoopsUnroll
