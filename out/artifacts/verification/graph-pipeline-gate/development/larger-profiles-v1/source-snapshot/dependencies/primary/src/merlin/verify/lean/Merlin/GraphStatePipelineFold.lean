/- Operational folds connected to W's ordered grouping proofs. All states may
contain failure and terminal world; only the proved absorbing representation
allows a total transition relation to be used. -/
import Merlin.GraphStatePipelineControl

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false
open Scalar GraphLoopsSegment GraphLoopsUnroll

theorem finish_idempotent {Value World Error : Type} (arity : Nat) (state : Outcome Value World Error) :
    finish arity (finish arity state) = finish arity state := by
  cases state with
  | error failure => rfl
  | ok state =>
    rcases state with ⟨values,world⟩
    by_cases size : values.length = arity <;> simp only [finish,size,↓reduceIte]

theorem finish_advance_finish {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (state : Outcome Value World Error) (index : Int) :
    finish arity (advance arity body (finish arity state) index) =
      finish arity (advance arity body state index) := by
  cases state with
  | error failure => rfl
  | ok state =>
    rcases state with ⟨values,world⟩
    by_cases size : values.length = arity <;> simp only [finish,advance,size,↓reduceIte]

def closedAdvance {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (state : Outcome Value World Error) (index : Int) : Outcome Value World Error :=
  finish arity (advance arity body state index)

/-- Immediate arity validation and absorbing errors are a representation of
loopEval, not a replacement evaluator whose correspondence is assumed. -/
theorem fold_closed {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (indices : List Int) (state : Outcome Value World Error) :
    indices.foldl (closedAdvance arity body) (finish arity state) =
      finish arity (indices.foldl (advance arity body) state) := by
  induction indices generalizing state with
  | nil => rfl
  | cons index later ih =>
    simp only [List.foldl_cons,closedAdvance,finish_advance_finish]
    exact ih (advance arity body state index)

theorem loop_closed {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (indices : List Int) (values : List Value) (world : World) :
    Scalar.loopEval arity body indices values world =
      indices.foldl (closedAdvance arity body) (finish arity (.ok (values,world))) := by
  rw [loop_fold, fold_closed]

def groupStep {State : Type} (transition : Nat → State → State) (step factor index : Nat)
    (before : State × List Nat) : State × List Nat :=
  let indices := progression index step factor
  (indices.foldl (fun state iv => transition iv state) before.1, before.2 ++ indices)

/-- Actual grouped state/log meaning is derived from the counted relation. -/
theorem groupBody_step {State : Type} (transition : Nat → State → State) (step factor index : Nat)
    (before after : State × List Nat) :
    GroupBody (Step transition) step factor index before after ↔
      Step (groupStep transition step factor) index before after := by
  constructor
  · rintro ⟨trace,run,log⟩
    obtain ⟨indices,folded⟩ := (repeat_fold transition step factor index before.1 after.1 trace).mp run
    rcases before with ⟨initial,startLog⟩
    rcases after with ⟨final,endLog⟩
    dsimp only at indices folded log
    subst trace
    dsimp only [Step,groupStep]
    exact Prod.ext folded log
  · intro same
    change after = groupStep transition step factor index before at same
    subst after
    refine ⟨progression index step factor, ?_, rfl⟩
    exact (repeat_fold transition step factor index _ _ _).mpr ⟨rfl,rfl⟩

theorem repeat_congr {State : Type} {first second : Nat → State → State → Prop}
    (same : ∀ index before after, first index before after ↔ second index before after)
    {step count lower : Nat} {initial final : State} {trace : List Nat}
    (run : Repeat first step count lower initial trace final) :
    Repeat second step count lower initial trace final := by
  induction run with
  | nil lower state => exact .nil lower state
  | cons event _ ih => exact .cons ((same _ _ _).mp event) ih

/-- W's grouping theorem entails equality of complete deterministic folds.
This is independent of trip count magnitude and does not expand body execution. -/
theorem groups_fold {State : Type} (transition : Nat → State → State)
    (lower step factor groups : Nat) (initial : State) :
    ((progression lower (step*factor) groups).foldl
      (fun state index => groupStep transition step factor index state) (initial,[])).1 =
      (progression lower step (groups*factor)).foldl (fun state index => transition index state) initial := by
  let bases := progression lower (step*factor) groups
  let final := bases.foldl (fun state index => groupStep transition step factor index state) (initial,[])
  have run : Repeat (Step (groupStep transition step factor)) (step*factor) groups lower (initial,[]) bases final :=
    (repeat_fold _ _ _ _ _ _ _).mpr ⟨rfl,rfl⟩
  have grouped := repeat_congr (fun index before after => (groupBody_step transition step factor index before after).symm) run
  obtain ⟨trace,original,_⟩ :=
    (grouped_repeat_iff (Step transition) step factor groups lower initial final.1 [] final.2).mp ⟨bases,grouped⟩
  obtain ⟨same,folded⟩ := (repeat_fold transition step (groups*factor) lower initial final.1 trace).mp original
  rw [same] at folded
  exact folded

/-- A total relation execution exists for any deterministic transition. This
includes absorbing failures, with zero additional external calls after failure. -/
theorem range_execution {State : Type} (transition : Nat → State → State)
    (upper step : Nat) (positive : 0 < step) (lower : Nat) (initial : State) :
    Execution (Step transition) upper step lower initial (scfRange lower upper step)
      ((scfRange lower upper step).foldl (fun state index => transition index state) initial) := by
  by_cases inside : lower < upper
  · rw [scfRange]
    simp only [inside,positive,and_self,↓reduceIte,List.foldl_cons]
    exact .more inside rfl (range_execution transition upper step positive (lower+step) (transition lower initial))
  · have stopped : upper ≤ lower := by omega
    rw [scfRange]
    simp only [inside,false_and,↓reduceIte,List.foldl_nil]
    exact .done initial stopped
termination_by upper-lower
decreasing_by omega

theorem execution_determined {State : Type} {transition : Nat → State → State}
    {upper step lower : Nat} (positive : 0 < step) {initial final : State} {trace : List Nat}
    (run : Execution (Step transition) upper step lower initial trace final) :
    trace = scfRange lower upper step ∧ final = trace.foldl (fun state index => transition index state) initial := by
  induction run with
  | @done lower state stopped =>
    constructor
    · rw [scfRange]; simp only [show ¬ (lower < upper ∧ 0 < step) by omega,↓reduceIte]
    · rfl
  | @more lower initial middle final later inside event remaining ih =>
    constructor
    · rw [scfRange]
      simp only [inside,positive,and_self,↓reduceIte,ih.1]
    · change middle = transition lower initial at event
      rw [List.foldl_cons, ← event]
      exact ih.2

end Merlin.GraphState.Pipeline
