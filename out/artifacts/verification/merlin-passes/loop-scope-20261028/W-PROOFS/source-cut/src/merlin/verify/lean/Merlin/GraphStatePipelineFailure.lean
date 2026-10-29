/- Failure-sensitive bridge for the unchanged sealed scalar evaluator.
A logical trace may visit later indices after failure, but its absorbing state
neither invokes the body nor changes the terminal error/world. This representation
is proved equivalent to early failure; no source-success premise is introduced. -/
import Merlin.GraphStateScalarContract
-- One shared library contains byte-identical sealed dependency objects.

namespace Merlin.GraphState.Pipeline
open Scalar
set_option autoImplicit false

/-- The final arity check is observable, including for a zero-trip loop. -/
def finish {Value World Error : Type} (arity : Nat) : Outcome Value World Error → Outcome Value World Error
  | .error failure => .error failure
  | .ok (values,world) =>
    if values.length = arity then .ok (values,world) else .error (.arity,world)

/-- An absorbing step checks current carries and executes at most one body.
It leaves malformed result arity for the next step/final check, exactly as loopEval. -/
def advance {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (before : Outcome Value World Error) (index : Int) : Outcome Value World Error :=
  match before with
  | .error failure => .error failure
  | .ok (values,world) =>
    if values.length = arity then body index values world else .error (.arity,world)

/-- Absorption preserves the complete terminal failure and does not invoke body. -/
theorem advance_failed {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (failure : Fault Error × World) (index : Int) :
    advance arity body (.error failure) index = .error failure := rfl

theorem fold_failed {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (failure : Fault Error × World) (indices : List Int) :
    indices.foldl (advance arity body) (.error failure) = .error failure := by
  induction indices with
  | nil => rfl
  | cons index later ih => exact ih

/-- Exact representation theorem, for arbitrary inputs/body/arity/errors/world.
This proves the failure-prefix semantics needed to use total relational W roots. -/
theorem fold_early_failure {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (indices : List Int) (initial : Outcome Value World Error) :
    finish arity (indices.foldl (advance arity body) initial) =
      initial.bind (fun state => Scalar.loopEval arity body indices state.1 state.2) := by
  induction indices generalizing initial with
  | nil =>
    cases initial with
    | error failure => rfl
    | ok state => cases state; rfl
  | cons index later ih =>
    cases initial with
    | error failure => simp only [fold_failed, finish, Except.bind]
    | ok state =>
      rcases state with ⟨values,world⟩
      simp only [List.foldl_cons, ih]
      by_cases h : values.length = arity
      · simp only [advance, h, ↓reduceIte, Except.bind, Scalar.loopEval]
      · simp only [advance, h, ↓reduceIte, Except.bind, Scalar.loopEval]

theorem loop_fold {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (indices : List Int) (values : List Value) (world : World) :
    Scalar.loopEval arity body indices values world =
      finish arity (indices.foldl (advance arity body) (.ok (values,world))) :=
  (fold_early_failure arity body indices (.ok (values,world))).symm

/-- Regrouping an absorbing fold preserves failure at any prefix. In particular
there is no assumption that a complete original body trace succeeds. -/
theorem fold_append {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (first second : List Int) (initial : Outcome Value World Error) :
    (first ++ second).foldl (advance arity body) initial =
      second.foldl (advance arity body) (first.foldl (advance arity body) initial) :=
  List.foldl_append

/-- The remainder of a failed prefix cannot mutate its error or terminal world. -/
theorem failure_prefix {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (first second : List Int) (initial : Outcome Value World Error) (failure : Fault Error × World)
    (failed : first.foldl (advance arity body) initial = .error failure) :
    finish arity ((first ++ second).foldl (advance arity body) initial) = .error failure := by
  rw [fold_append, failed, fold_failed]
  rfl

end Merlin.GraphState.Pipeline
