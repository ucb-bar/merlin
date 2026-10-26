/- Actual ordered unrolling over the existing partial/effectful SCF semantics.
Repeat is an exact sequence of ORIGINAL body occurrences, preserving every
intermediate carry/state binding. No source success or arithmetic law is inferred. -/
import Merlin.GraphLoopsSegmentState

set_option autoImplicit false

namespace Merlin.GraphLoopsUnroll

open GraphLoopsSegment

inductive Repeat {σ : Type} (body : Nat → σ → σ → Prop) (step : Nat) :
    Nat → Nat → σ → List Nat → σ → Prop where
  | nil (lower : Nat) (state : σ) : Repeat body step 0 lower state [] state
  | cons {count lower : Nat} {initial middle final : σ} {rest : List Nat}
      (event : body lower initial middle)
      (remaining : Repeat body step count (lower + step) middle rest final) :
      Repeat body step (count + 1) lower initial (lower :: rest) final

def CountCorrect (lower upper step : Nat) : Nat → Prop
  | 0 => upper ≤ lower
  | count + 1 => lower + step * count < upper ∧ upper ≤ lower + step * (count + 1)

instance (lower upper step count : Nat) : Decidable (CountCorrect lower upper step count) := by
  cases count <;> unfold CountCorrect <;> infer_instance

private theorem next_end (lower step count : Nat) :
    lower + step * (count + 1) = (lower + step) + step * count := by
  simp only [Nat.mul_add, Nat.mul_one]
  omega

theorem CountCorrect.live {lower upper step count : Nat}
    (correct : CountCorrect lower upper step (count + 1)) : lower < upper := by
  unfold CountCorrect at correct
  omega

theorem CountCorrect.next {lower upper step count : Nat}
    (correct : CountCorrect lower upper step (count + 1)) : CountCorrect (lower + step) upper step count := by
  cases count with
  | zero => simpa only [CountCorrect, Nat.mul_zero, Nat.add_zero, Nat.zero_add, Nat.mul_one] using correct.2
  | succ count =>
    simp only [CountCorrect] at correct ⊢
    simpa only [next_end] using correct

/-- The checked arithmetic count is tied to the existing complete SCF relation.
This proof is parametric and never constructs a trip-count-sized certificate. -/
theorem repeat_iff_execution {σ : Type} (body : Nat → σ → σ → Prop) (upper step count : Nat) :
    ∀ lower initial final trace, CountCorrect lower upper step count →
      (Repeat body step count lower initial trace final ↔ Execution body upper step lower initial trace final) := by
  induction count with
  | zero =>
    intro lower initial final trace correct
    constructor
    · intro repeated; cases repeated; exact .done _ correct
    · intro executed
      obtain ⟨rfl, rfl⟩ := executed.stopped correct
      exact .nil _ _
  | succ count induction =>
    intro lower initial final trace correct
    constructor
    · intro repeated
      cases repeated with
      | cons event remaining =>
        exact .more correct.live event ((induction _ _ _ _ correct.next).mp remaining)
    · intro executed
      cases executed with
      | done _ stopped => have := correct.live; omega
      | more _ event remaining =>
        exact .cons event ((induction _ _ _ _ correct.next).mpr remaining)

theorem Repeat.length {σ : Type} {body : Nat → σ → σ → Prop} {step count lower : Nat}
    {initial final : σ} {trace : List Nat} (run : Repeat body step count lower initial trace final) :
    trace.length = count := by
  induction run with
  | nil => rfl
  | cons _ _ induction => simpa only [List.length_cons] using congrArg Nat.succ induction

/-- Rejoin adjacent exact sequences without reassociating their body operations. -/
theorem Repeat.append {σ : Type} {body : Nat → σ → σ → Prop} {step firstCount lower : Nat}
    {initial middle : σ} {first : List Nat} (left : Repeat body step firstCount lower initial first middle) :
    ∀ restCount final rest, Repeat body step restCount (lower + step * firstCount) middle rest final →
      Repeat body step (firstCount + restCount) lower initial (first ++ rest) final := by
  induction left with
  | nil lower state =>
    intro restCount final rest right
    simpa only [Nat.mul_zero, Nat.add_zero, Nat.zero_add, List.nil_append] using right
  | @cons count lower initial next middle trace event remaining induction =>
    intro restCount final rest right
    rw [next_end] at right
    have result := Repeat.cons event (induction restCount final rest right)
    simpa only [List.cons_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using result

theorem Repeat.split {σ : Type} (body : Nat → σ → σ → Prop) (step firstCount : Nat) :
    ∀ restCount lower initial final trace, Repeat body step (firstCount + restCount) lower initial trace final →
      ∃ middle first rest, trace = first ++ rest ∧ Repeat body step firstCount lower initial first middle ∧
        Repeat body step restCount (lower + step * firstCount) middle rest final := by
  induction firstCount with
  | zero =>
    intro restCount lower initial final trace run
    exact ⟨initial, [], trace, rfl, .nil _ _, by simpa only [Nat.mul_zero, Nat.add_zero, Nat.zero_add] using run⟩
  | succ count induction =>
    intro restCount lower initial final trace run
    have adjusted : Repeat body step ((count + restCount) + 1) lower initial trace final := by
      simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using run
    cases adjusted with
    | @cons _ _ _ next _ restTrace event remaining =>
      obtain ⟨middle, first, rest, same, left, right⟩ := induction restCount (lower + step) next final restTrace remaining
      refine ⟨middle, lower :: first, rest, ?_, .cons event left, ?_⟩
      · simpa only [List.cons_append] using congrArg (List.cons lower) same
      · simpa only [next_end] using right

end Merlin.GraphLoopsUnroll
