/- Ordered SCF strip-mining, independent of the counted Kernel model.
The body is an admitted relation over the ENTIRE carried value and modeled state.
It may be partial or effectful: these theorems infer neither purity nor success.
The actual source-to-model/SSA correspondence remains a separate checked boundary.
Foundation's orderedSegments/rangeSplit motivate this specialization, but the full
LeanMLIR route is not selected; existing StateEffects state is reused directly. -/
import Merlin.StateEffects
import Merlin.ResourceGeometry

set_option autoImplicit false

namespace Merlin.GraphLoopsSegment

/-- Each body occurrence uses its actual IV and exact incoming/outgoing state. -/
inductive Execution {σ : Type} (body : Nat → σ → σ → Prop) (upper step : Nat) :
    Nat → σ → List Nat → σ → Prop where
  | done {lower : Nat} (state : σ) (stopped : upper ≤ lower) :
      Execution body upper step lower state [] state
  | more {lower : Nat} {initial middle final : σ} {trace : List Nat}
      (inside : lower < upper) (transition : body lower initial middle)
      (remaining : Execution body upper step (lower + step) middle trace final) :
      Execution body upper step lower initial (lower :: trace) final

/-- Exact semantics of the generated outer loop and its bounded inner loop. -/
inductive Segmented {σ : Type} (body : Nat → σ → σ → Prop) (upper step tile : Nat) :
    Nat → σ → List Nat → σ → Prop where
  | done {lower : Nat} (state : σ) (stopped : upper ≤ lower) :
      Segmented body upper step tile lower state [] state
  | more {lower : Nat} {initial middle final : σ} {first rest : List Nat}
      (inside : lower < upper)
      (inner : Execution body (min (lower + step * tile) upper) step lower initial first middle)
      (remaining : Segmented body upper step tile (lower + step * tile) middle rest final) :
      Segmented body upper step tile lower initial (first ++ rest) final

theorem Execution.stopped {σ : Type} {body : Nat → σ → σ → Prop} {upper step lower : Nat}
    {initial final : σ} {trace : List Nat} (run : Execution body upper step lower initial trace final)
    (ended : upper ≤ lower) : trace = [] ∧ final = initial := by
  cases run with
  | done => exact ⟨rfl, rfl⟩
  | more inside _ _ => omega

private theorem block_end (lower step tile : Nat) :
    lower + step * (tile + 1) = (lower + step) + step * tile := by
  simp only [Nat.mul_add, Nat.mul_one]
  omega

/-- Rejoin the exact bounded inner trace with the next outer position. This is
the operational counterpart of rangeSplit; it does not reassociate any body step. -/
theorem join_aligned {σ : Type} (body : Nat → σ → σ → Prop) (upper step : Nat) (positive : 0 < step) :
    ∀ tile lower initial middle final first rest,
      Execution body (min (lower + step * tile) upper) step lower initial first middle →
      Execution body upper step (lower + step * tile) middle rest final →
      Execution body upper step lower initial (first ++ rest) final := by
  intro tile
  induction tile with
  | zero =>
    intro lower initial middle final first rest inner remaining
    have ended : min (lower + step * 0) upper ≤ lower := by
      simpa only [Nat.mul_zero, Nat.add_zero] using Nat.min_le_left lower upper
    obtain ⟨rfl, rfl⟩ := inner.stopped ended
    simpa only [Nat.mul_zero, Nat.add_zero, List.nil_append] using remaining
  | succ tile induction =>
    intro lower initial middle final first rest inner remaining
    by_cases ended : upper ≤ lower
    · have innerEnded : min (lower + step * (tile + 1)) upper ≤ lower :=
        Nat.le_trans (Nat.min_le_right _ _) ended
      obtain ⟨rfl, rfl⟩ := inner.stopped innerEnded
      have nextEnded : upper ≤ lower + step * (tile + 1) := Nat.le_trans ended (Nat.le_add_right _ _)
      obtain ⟨rfl, rfl⟩ := remaining.stopped nextEnded
      exact .done _ ended
    · have live : lower < upper := Nat.lt_of_not_ge ended
      have advances : 0 < step * (tile + 1) := Nat.mul_pos positive (Nat.succ_pos tile)
      have innerLive : lower < min (lower + step * (tile + 1)) upper := by omega
      cases inner with
      | done _ stopped => omega
      | @more _ _ next _ firstTail _ transition restInner =>
        rw [block_end] at restInner remaining
        exact .more live transition (induction (lower + step) next middle final firstTail rest restInner remaining)

/-- Every complete generated nested execution is a complete source execution,
with identical ordered IV trace and final carry/state. -/
theorem segmented_refines {σ : Type} {body : Nat → σ → σ → Prop} {upper step tile lower : Nat}
    (positive : 0 < step) {initial final : σ} {trace : List Nat}
    (run : Segmented body upper step tile lower initial trace final) :
    Execution body upper step lower initial trace final := by
  induction run with
  | done state stopped => exact .done state stopped
  | more _ inner _ induction => exact join_aligned _ _ _ positive _ _ _ _ _ _ _ inner induction

/-- A complete source execution splits at the actual aligned next-outer IV.
The tail may be empty; no extra body evaluation is introduced in that case. -/
theorem split_aligned {σ : Type} (body : Nat → σ → σ → Prop) (upper step : Nat) (positive : 0 < step) :
    ∀ tile lower initial final trace, Execution body upper step lower initial trace final →
      ∃ middle first rest, trace = first ++ rest ∧
        Execution body (min (lower + step * tile) upper) step lower initial first middle ∧
        Execution body upper step (lower + step * tile) middle rest final := by
  intro tile
  induction tile with
  | zero =>
    intro lower initial final trace run
    refine ⟨initial, [], trace, rfl, ?_, ?_⟩
    · exact .done initial (by simpa only [Nat.mul_zero, Nat.add_zero] using Nat.min_le_left lower upper)
    · simpa only [Nat.mul_zero, Nat.add_zero] using run
  | succ tile induction =>
    intro lower initial final trace run
    by_cases ended : upper ≤ lower
    · obtain ⟨rfl, rfl⟩ := run.stopped ended
      refine ⟨_, [], [], rfl, .done _ ?_, .done _ ?_⟩
      · exact Nat.le_trans (Nat.min_le_right _ _) ended
      · exact Nat.le_trans ended (Nat.le_add_right _ _)
    · have live : lower < upper := Nat.lt_of_not_ge ended
      have advances : 0 < step * (tile + 1) := Nat.mul_pos positive (Nat.succ_pos tile)
      have innerLive : lower < min (lower + step * (tile + 1)) upper := by omega
      cases run with
      | done _ stopped => omega
      | @more _ _ next _ remainingTrace _ transition remaining =>
        obtain ⟨middle, first, rest, equal, inner, suffix⟩ :=
          induction (lower + step) next final remainingTrace remaining
        refine ⟨middle, lower :: first, rest, ?_, ?_, ?_⟩
        · simpa only [List.cons_append] using congrArg (List.cons lower) equal
        · rw [block_end]
          exact .more (by simpa only [← block_end] using innerLive) transition inner
        · simpa only [block_end] using suffix

theorem Execution.live_trace {σ : Type} {body : Nat → σ → σ → Prop} {upper step lower : Nat}
    {initial final : σ} {trace : List Nat} (run : Execution body upper step lower initial trace final)
    (inside : lower < upper) : trace ≠ [] := by
  cases run with
  | done _ stopped => omega
  | more => simp

/-- Source and segmented executions have exactly the same possible complete
traces and final carry/state, even when the admitted body relation is partial. -/
theorem source_refines {σ : Type} (body : Nat → σ → σ → Prop) (upper step tile : Nat)
    (positiveStep : 0 < step) (positiveTile : 0 < tile) :
    ∀ lower initial final trace, Execution body upper step lower initial trace final →
      Segmented body upper step tile lower initial trace final := by
  have byLength : ∀ count, ∀ lower initial final trace, trace.length = count →
      Execution body upper step lower initial trace final → Segmented body upper step tile lower initial trace final := by
    intro count
    induction count using Nat.strongRecOn with
    | ind count induction =>
      intro lower initial final trace length run
      by_cases ended : upper ≤ lower
      · obtain ⟨rfl, rfl⟩ := run.stopped ended
        exact .done _ ended
      · have live : lower < upper := Nat.lt_of_not_ge ended
        obtain ⟨middle, first, rest, equal, inner, suffix⟩ :=
          split_aligned body upper step positiveStep tile lower initial final trace run
        have advances : 0 < step * tile := Nat.mul_pos positiveStep positiveTile
        have innerLive : lower < min (lower + step * tile) upper := by omega
        have nonempty := inner.live_trace innerLive
        have firstPositive : 0 < first.length := List.length_pos_iff.mpr nonempty
        have shorter : rest.length < count := by
          rw [equal, List.length_append] at length
          omega
        have next := induction rest.length shorter (lower + step * tile) middle final rest rfl suffix
        rw [equal]
        exact .more live inner next
  intro lower initial final trace run
  exact byLength trace.length lower initial final trace rfl run

theorem complete_trace_iff {σ : Type} (body : Nat → σ → σ → Prop) (upper step tile lower : Nat)
    (positiveStep : 0 < step) (positiveTile : 0 < tile) (initial final : σ) (trace : List Nat) :
    Execution body upper step lower initial trace final ↔
      Segmented body upper step tile lower initial trace final := by
  exact ⟨source_refines _ _ _ _ positiveStep positiveTile _ _ _ _, segmented_refines positiveStep⟩

end Merlin.GraphLoopsSegment
