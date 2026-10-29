/- Source-only candidate. No native validation has been performed.
Ordered bulk plus remainder in the unchanged Scalar operational interpretation.
The body equation below is a local obligation to close using actual decoded SSA;
it is never an external contract or a premise of the emitted complete-entry root. -/
import Merlin.GraphStatePipelineLoops

namespace Merlin.GraphState.Pipeline.Tail
set_option autoImplicit false
open Scalar GraphLoopsUnroll

theorem progression_append (lower step first rest : Nat) :
    progression lower step (first + rest) =
      progression lower step first ++ progression (lower + step * first) step rest := by
  induction first generalizing lower with
  | zero => simp only [Nat.zero_add, progression, List.nil_append, Nat.mul_zero, Nat.add_zero]
  | succ first ih =>
    simpa only [Nat.succ_add, progression, List.cons_append, Nat.mul_succ,
      Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
      congrArg (List.cons lower) (ih (lower+step))

/-- Splitting never executes the second interval after a first-interval failure.
The complete fault and terminal world, including arity failures, are retained. -/
theorem loop_append {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (first rest : List Int) (values : List Value) (world : World) :
    loopEval arity body (first ++ rest) values world =
      (loopEval arity body first values world).bind
        (fun result => loopEval arity body rest result.1 result.2) := by
  induction first generalizing values world with
  | nil =>
    by_cases size : values.length = arity
    · simp only [List.nil_append, loopEval, size, ↓reduceIte, Except.bind]
    · cases rest <;> simp only [List.nil_append, loopEval, size, ↓reduceIte, Except.bind]
  | cons index later ih =>
    by_cases size : values.length = arity
    · simp only [List.cons_append, loopEval, size, ↓reduceIte]
      cases result : body index values world with
      | error failure => rfl
      | ok state => exact ih state.1 state.2
    · simp only [List.cons_append, loopEval, size, ↓reduceIte, Except.bind]

/-- Structural counted split plus sealed W grouping. No trip-sized computation,
source-success premise, error commutativity, or arithmetic reassociation. -/
theorem ordered_bulk_tail {Value World Error : Type} (arity : Nat)
    (original bulk : Int → List Value → World → Outcome Value World Error)
    (lower upper step factor groups remainder : Nat)
    (positive : 0 < step) (positiveFactor : 0 < factor)
    (counted : CountCorrect lower upper step (groups * factor + remainder))
    (bodyChecked : ∀ index values world, values.length = arity →
      bulk (Int.ofNat index) values world =
        loopEval arity original ((progression index step factor).map Int.ofNat) values world)
    (values : List Value) (world : World) :
    loopEval arity original (indexIterations lower upper step) values world =
      (loopEval arity bulk
        (indexIterations (lower : Int) ((lower + (step * factor) * groups : Nat) : Int)
          ((step * factor : Nat) : Int)) values world).bind
        (fun result => loopEval arity original
          (indexIterations ((lower + (step * factor) * groups : Nat) : Int) upper step)
          result.1 result.2) := by
  have cutoff : lower + step * (groups * factor) = lower + (step * factor) * groups := by
    simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
  have prefixCount := CountCorrect.aligned lower step (groups * factor) positive
  have suffixCount := CountCorrect.drop counted
  have split : indexIterations lower upper step =
      indexIterations (lower : Int) ((lower + step * (groups * factor) : Nat) : Int) step ++
      indexIterations ((lower + step * (groups * factor) : Nat) : Int) upper step := by
    rw [indexIterations_count positive counted, indexIterations_count positive prefixCount,
      indexIterations_count positive suffixCount, progression_append, List.map_append]
  rw [split, loop_append,
    unroll_loop arity original bulk lower (lower + step * (groups * factor)) step factor groups
      positive positiveFactor prefixCount bodyChecked, cutoff]

/-- Sequential continuation congruence retains arbitrary observers, failures,
and terminal worlds. It is used only after both actual loop equations close. -/
theorem continue_same {Value World Error : Type}
    (first second : Outcome Value World Error)
    (left right : List Value → World → Outcome Value World Error)
    (loops : first = second) (suffix : ∀ values world, left values world = right values world) :
    first.bind (fun result => left result.1 result.2) =
      second.bind (fun result => right result.1 result.2) := by
  rw [loops]
  cases second with
  | error failure => rfl
  | ok state => exact suffix state.1 state.2

end Merlin.GraphState.Pipeline.Tail
