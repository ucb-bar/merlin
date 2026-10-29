/- Failure-sensitive operational loop composition. The local body equations
below are proof obligations discharged against actual checked SSA syntax, not
external operation assumptions or assumed whole-program equalities. -/
import Merlin.GraphStatePipelineSegments
import Merlin.GraphStateScalarSound

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false
open Scalar GraphLoopsSegment GraphLoopsUnroll

abbrev Packed (Value World Error : Type) (arity : Nat) :=
  {state : Outcome Value World Error // finish arity state = state}

def packedInitial {Value World Error : Type} (arity : Nat) (values : List Value) (world : World) :
    Packed Value World Error arity := ⟨finish arity (.ok (values,world)),finish_idempotent _ _⟩

def packedStep {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (index : Nat) (before : Packed Value World Error arity) : Packed Value World Error arity :=
  ⟨closedAdvance arity body before.1 (Int.ofNat index),finish_idempotent _ _⟩

/-- Every successful packed carry has the checked length. An error remains
an error, with no assumption of successful source/body execution. -/
theorem packed_arity {Value World Error : Type} {arity : Nat}
    (state : Packed Value World Error arity) {values : List Value} {world : World}
    (success : state.1 = .ok (values,world)) : values.length = arity := by
  have fixed := state.2
  rw [success] at fixed
  by_cases size : values.length = arity
  · exact size
  · simp only [finish,size,↓reduceIte] at fixed
    cases fixed

/-- Projection of the subtype fold is the exact absorbing fold. -/
theorem packed_fold {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error) (indices : List Nat)
    (state : Packed Value World Error arity) :
    (indices.foldl (fun s i => packedStep arity body i s) state).1 =
      (indices.map Int.ofNat).foldl (closedAdvance arity body) state.1 := by
  induction indices generalizing state with
  | nil => rfl
  | cons index later ih => exact ih (packedStep arity body index state)

/-- The packed fold is proved equivalent to the unchanged early-failure loop. -/
theorem packed_run {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error) (indices : List Nat)
    (state : Packed Value World Error arity) :
    (indices.foldl (fun s i => packedStep arity body i s) state).1 =
      state.1.bind (fun result => Scalar.loopEval arity body (indices.map Int.ofNat) result.1 result.2) := by
  rw [packed_fold]
  have fixed := congrArg (fun value => (indices.map Int.ofNat).foldl (closedAdvance arity body) value) state.2
  exact fixed.symm.trans ((fold_closed arity body _ state.1).trans (fold_early_failure arity body _ state.1))

/-- A proved actual SSA block representing an original-body interval induces
exactly the corresponding packed transition, including first failure/world. -/
theorem packed_block {Value World Error : Type} (arity : Nat)
    (original block : Int → List Value → World → Outcome Value World Error)
    (indices : Nat → List Nat)
    (bodyChecked : ∀ index values world, values.length = arity →
      block (Int.ofNat index) values world =
        Scalar.loopEval arity original ((indices index).map Int.ofNat) values world)
    (index : Nat) (state : Packed Value World Error arity) :
    packedStep arity block index state =
      (indices index).foldl (fun s i => packedStep arity original i s) state := by
  apply Subtype.ext
  rw [packed_run]
  cases value : state.1 with
  | error failure => simp only [packedStep,closedAdvance,advance,value,finish,Except.bind]
  | ok result =>
    rcases result with ⟨values,world⟩
    have size := packed_arity state value
    simp only [packedStep,closedAdvance,advance,value,size,↓reduceIte,Except.bind]
    rw [bodyChecked index values world size]
    cases executed : Scalar.loopEval arity original ((indices index).map Int.ofNat) values world with
    | error failure => rfl
    | ok result =>
      have resultSize := loopEval_arity arity original _ _ _ executed
      simp only [finish,resultSize,↓reduceIte]

theorem fold_pointwise {State : Type} (first second : Nat → State → State)
    (same : ∀ index state, first index state = second index state) (indices : List Nat) (initial : State) :
    indices.foldl (fun state index => first index state) initial =
      indices.foldl (fun state index => second index state) initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index later ih => simp only [List.foldl_cons,same,ih]

/-- Operational unroll preservation from the exact local cloned-body proof
and existing W grouping root. Bound arithmetic is checked separately by W. -/
theorem unroll_loop {Value World Error : Type} (arity : Nat)
    (original bulk : Int → List Value → World → Outcome Value World Error)
    (lower upper step factor groups : Nat) (positive : 0 < step) (positiveFactor : 0 < factor)
    (counted : CountCorrect lower upper step (groups*factor))
    (bodyChecked : ∀ index values world, values.length = arity →
      bulk (Int.ofNat index) values world =
        Scalar.loopEval arity original ((progression index step factor).map Int.ofNat) values world)
    (values : List Value) (world : World) :
    Scalar.loopEval arity original (indexIterations lower upper step) values world =
      Scalar.loopEval arity bulk
        (indexIterations (lower : Int) ((lower+(step*factor)*groups : Nat) : Int) ((step*factor : Nat) : Int)) values world := by
  have originalRange := indexIterations_count positive counted
  have bulkRange := indexIterations_count (Nat.mul_pos positive positiveFactor)
    (CountCorrect.aligned lower (step*factor) groups (Nat.mul_pos positive positiveFactor))
  let initial : Packed Value World Error arity := packedInitial arity values world
  have originalProjection : Scalar.loopEval arity original (indexIterations lower upper step) values world =
      ((progression lower step (groups*factor)).foldl (fun s i => packedStep arity original i s) initial).1 := by
    rw [originalRange,packed_fold]
    exact loop_closed arity original _ values world
  have bulkProjection : Scalar.loopEval arity bulk
      (indexIterations (lower : Int) ((lower+(step*factor)*groups : Nat) : Int) ((step*factor : Nat) : Int)) values world =
      ((progression lower (step*factor) groups).foldl (fun s i => packedStep arity bulk i s) initial).1 := by
    rw [bulkRange,packed_fold]
    exact loop_closed arity bulk _ values world
  rw [originalProjection,bulkProjection]
  have replace := fold_pointwise (packedStep arity bulk)
    (fun index state => (progression index step factor).foldl (fun s i => packedStep arity original i s) state)
    (packed_block arity original bulk (fun index => progression index step factor) bodyChecked)
    (progression lower (step*factor) groups) initial
  rw [replace, groups_plain]

/-- Operational segmentation preservation, with explicit actual outer-body
correspondence to the inner finite SCF interval. Failure/world is part of state. -/
theorem segment_loop {Value World Error : Type} (arity : Nat)
    (original outer : Int → List Value → World → Outcome Value World Error)
    (lower upper step tile : Nat) (positive : 0 < step) (positiveTile : 0 < tile)
    (bodyChecked : ∀ index values world, values.length = arity →
      outer (Int.ofNat index) values world = Scalar.loopEval arity original
        ((scfRange index (min (index+step*tile) upper) step).map Int.ofNat) values world)
    (values : List Value) (world : World) :
    Scalar.loopEval arity original ((scfRange lower upper step).map Int.ofNat) values world =
      Scalar.loopEval arity outer ((scfRange lower upper (step*tile)).map Int.ofNat) values world := by
  let initial : Packed Value World Error arity := packedInitial arity values world
  rw [loop_closed,loop_closed]
  change ((scfRange lower upper step).map Int.ofNat).foldl (closedAdvance arity original) initial.1 =
    ((scfRange lower upper (step*tile)).map Int.ofNat).foldl (closedAdvance arity outer) initial.1
  rw [← packed_fold arity original _ initial,← packed_fold arity outer _ initial]
  have replace := fold_pointwise (packedStep arity outer)
    (fun index state => (scfRange index (min (index+step*tile) upper) step).foldl
      (fun s i => packedStep arity original i s) state)
    (packed_block arity original outer (fun index => scfRange index (min (index+step*tile) upper) step) bodyChecked)
    (scfRange lower upper (step*tile)) initial
  rw [replace,segments_plain _ upper step tile lower positive positiveTile]

end Merlin.GraphState.Pipeline
