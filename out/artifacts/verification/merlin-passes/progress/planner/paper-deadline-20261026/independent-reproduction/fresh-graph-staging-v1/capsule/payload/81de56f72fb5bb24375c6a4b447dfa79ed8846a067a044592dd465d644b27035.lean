/- Explicit finite nonnegative SCF control and its connection to the sealed W
relation. The evaluator determines IVs from bounds/stride; no iteration callback
agreement or successful-body hypothesis is used in these bridges. -/
import Merlin.GraphStatePipelineFailure
import Merlin.GraphLoopsUnrollGrouping

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false
open Scalar GraphLoopsSegment GraphLoopsUnroll

/-- Static nonnegative SCF iteration, including zero-trip and stride overshoot.
Invalid/nonpositive controls are outside the admitted checked domain. -/
def scfRange (lower upper step : Nat) : List Nat :=
  if lower < upper ∧ 0 < step then
    lower :: scfRange (lower+step) upper step
  else []
termination_by upper-lower
decreasing_by omega

/-- Exact counted progression. It is mathematical data, not a trip-sized
certificate: the preservation proofs recurse on the symbolic count. -/
def progression (lower step : Nat) : Nat → List Nat
  | 0 => []
  | count+1 => lower :: progression (lower+step) step count

theorem range_count {lower upper step count : Nat} (positive : 0 < step)
    (correct : CountCorrect lower upper step count) :
    scfRange lower upper step = progression lower step count := by
  induction count generalizing lower with
  | zero =>
    have stopped : upper ≤ lower := correct
    rw [scfRange]
    simp only [show ¬ (lower < upper ∧ 0 < step) by omega, ↓reduceIte, progression]
  | succ count ih =>
    rw [scfRange]
    simp only [show lower < upper ∧ 0 < step from ⟨correct.live,positive⟩, ↓reduceIte, progression]
    exact congrArg (List.cons lower) (ih correct.next)

/-- Concrete value-level iteration selection for the supported index domain.
Negative controls/nonpositive stride are refused by the checker, not licensed
as alternate semantics. This total fallback does not establish their behavior. -/
def indexIterations (lower upper step : Int) : List Int :=
  if 0 ≤ lower ∧ 0 ≤ upper ∧ 0 < step then
    (scfRange lower.toNat upper.toNat step.toNat).map Int.ofNat
  else []

theorem indexIterations_count {lower upper step count : Nat}
    (positive : 0 < step) (correct : CountCorrect lower upper step count) :
    indexIterations lower upper step = (progression lower step count).map Int.ofNat := by
  have pos : (0 : Int) < (step : Int) := by omega
  simp only [indexIterations, Int.natCast_nonneg, pos, and_self, ↓reduceIte, Int.toNat_natCast]
  rw [range_count positive correct]

/-- Deterministic total transition used only after the failure representation
has been proved. It does not turn a raw partial body into successful execution. -/
def Step {State : Type} (transition : Nat → State → State) (index : Nat) (before after : State) : Prop :=
  after = transition index before

/-- W's repeated relation has exactly the evaluator's trace and folded state. -/
theorem repeat_fold {State : Type} (transition : Nat → State → State) (step count lower : Nat)
    (initial final : State) (trace : List Nat) :
    Repeat (Step transition) step count lower initial trace final ↔
      trace = progression lower step count ∧ final = trace.foldl (fun state index => transition index state) initial := by
  induction count generalizing lower initial trace with
  | zero =>
    constructor
    · intro run; cases run; exact ⟨rfl,rfl⟩
    · rintro ⟨rfl,rfl⟩; exact .nil _ _
  | succ count ih =>
    constructor
    · intro run
      cases run with
      | cons event remaining =>
        obtain ⟨same,folded⟩ := (ih _ _ _).mp remaining
        constructor
        · exact congrArg (List.cons lower) same
        · dsimp only [Step] at event
          rw [event] at folded
          simpa only [List.foldl_cons] using folded
    · rintro ⟨rfl,folded⟩
      refine .cons rfl ((ih _ _ _).mpr ⟨rfl,?_⟩)
      exact folded

/-- Complete operational bridge to W's SCF relation for arbitrary state,
including the absorbing error/world state. The trace and final state are derived. -/
theorem execution_fold {State : Type} (transition : Nat → State → State)
    {upper step count lower : Nat} (positive : 0 < step)
    (correct : CountCorrect lower upper step count) (initial final : State) (trace : List Nat) :
    Execution (Step transition) upper step lower initial trace final ↔
      trace = scfRange lower upper step ∧ final = trace.foldl (fun state index => transition index state) initial := by
  rw [← repeat_iff_execution _ upper step count lower initial final trace correct,
    repeat_fold, range_count positive correct]

/-- The sealed early-failure scalar loop is exactly the final projection of
W's deterministic outcome-state relation. It preserves errors/terminal worlds
without assuming any raw body success or total external computation. -/
theorem scalar_execution {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    {lower upper step count : Nat} (positive : 0 < step)
    (correct : CountCorrect lower upper step count) (values : List Value) (world : World) :
    ∃ final, Execution
      (Step (fun index state => advance arity body state (Int.ofNat index))) upper step lower
      (.ok (values,world)) (scfRange lower upper step) final ∧
      Scalar.loopEval arity body (indexIterations lower upper step) values world = finish arity final := by
  let final := (scfRange lower upper step).foldl
    (fun state index => advance arity body state (Int.ofNat index)) (.ok (values,world))
  refine ⟨final, (execution_fold _ positive correct _ _ _).mpr ⟨rfl,rfl⟩, ?_⟩
  rw [indexIterations_count positive correct, loop_fold]
  simp only [final, List.foldl_map, range_count positive correct]

end Merlin.GraphState.Pipeline
