/- Derived scalar/SCF operational grouping and segmentation folds, through
unchanged W relational roots. No success premise: State can include a fault
and the complete terminal world with a proved absorbing transition. -/
import Merlin.GraphStatePipelineFold

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false
open GraphLoopsSegment GraphLoopsUnroll

theorem logged_first {State : Type} (transition : Nat → State → State)
    (occurrences : Nat → List Nat) (indices : List Nat) (initial : State) (history : List Nat) :
    (indices.foldl (fun state index => (transition index state.1,state.2++occurrences index)) (initial,history)).1 =
      indices.foldl (fun state index => transition index state) initial := by
  induction indices generalizing initial history with
  | nil => rfl
  | cons index later ih => exact ih (transition index initial) (history++occurrences index)

theorem groups_plain {State : Type} (transition : Nat → State → State)
    (lower step factor groups : Nat) (initial : State) :
    (progression lower (step*factor) groups).foldl
      (fun state index => (progression index step factor).foldl (fun s iv => transition iv s) state) initial =
      (progression lower step (groups*factor)).foldl (fun state index => transition index state) initial := by
  have grouped := groups_fold transition lower step factor groups initial
  rw [← grouped]
  exact (logged_first _ _ _ _ _).symm

def segmentStep {State : Type} (transition : Nat → State → State) (upper step tile index : Nat)
    (before : State × List Nat) : State × List Nat :=
  let indices := scfRange index (min (index+step*tile) upper) step
  (indices.foldl (fun state iv => transition iv state) before.1, before.2++indices)

/-- The complete actual outer iteration corresponds to an inner SCF execution
and the same state. Its complete ordered trace is derived, not hypothesized. -/
theorem outer_segmented {State : Type} (transition : Nat → State → State)
    (upper step tile : Nat) (positive : 0 < step) {lower : Nat}
    {initial final : State × List Nat} {bases : List Nat}
    (run : Execution (Step (segmentStep transition upper step tile)) upper (step*tile) lower initial bases final) :
    ∃ trace, Segmented (Step transition) upper step tile lower initial.1 trace final.1 ∧
      final.2 = initial.2++trace := by
  induction run with
  | @done lower state stopped => exact ⟨[],.done _ stopped,by simp⟩
  | @more lower initial middle final later inside event remaining ih =>
    change middle = segmentStep transition upper step tile lower initial at event
    subst middle
    obtain ⟨trace,rest,log⟩ := ih
    refine ⟨scfRange lower (min (lower+step*tile) upper) step ++ trace,
      .more inside (range_execution transition _ step positive lower initial.1) rest, ?_⟩
    simpa only [segmentStep,List.append_assoc] using log

/-- Exact nested finite-fold correspondence established through W's full-state
segmentation theorem, without equating raw SSA stacks or assuming source success. -/
theorem segments_plain {State : Type} (transition : Nat → State → State)
    (upper step tile lower : Nat) (positive : 0 < step) (positiveTile : 0 < tile) (initial : State) :
    (scfRange lower upper (step*tile)).foldl
      (fun state index => (scfRange index (min (index+step*tile) upper) step).foldl
        (fun s iv => transition iv s) state) initial =
      (scfRange lower upper step).foldl (fun state index => transition index state) initial := by
  let bases := scfRange lower upper (step*tile)
  let final := bases.foldl (fun state index => segmentStep transition upper step tile index state) (initial,[])
  have outer := range_execution (segmentStep transition upper step tile) upper (step*tile)
    (Nat.mul_pos positive positiveTile) lower (initial,[])
  obtain ⟨trace,nested,_⟩ := outer_segmented transition upper step tile positive outer
  have original := segmented_refines positive nested
  obtain ⟨same,folded⟩ := execution_determined positive original
  rw [same] at folded
  have projection : final.1 = bases.foldl
      (fun state index => (scfRange index (min (index+step*tile) upper) step).foldl
        (fun s iv => transition iv s) state) initial := by
    exact logged_first
      (fun index state => (scfRange index (min (index+step*tile) upper) step).foldl
        (fun s iv => transition iv s) state)
      (fun index => scfRange index (min (index+step*tile) upper) step) bases initial []
  exact projection.symm.trans folded

end Merlin.GraphState.Pipeline
