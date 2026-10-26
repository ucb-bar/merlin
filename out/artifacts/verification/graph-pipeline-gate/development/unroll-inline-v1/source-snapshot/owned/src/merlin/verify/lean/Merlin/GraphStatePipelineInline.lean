/- Source-only empty/full-inline operational bridge. No replacement evaluator,
body-agreement axiom or source-success assumption. Instantiated queries close
the finite actual syntax equations in the unchanged Scalar evaluator. -/
import Merlin.GraphStatePipelineControl
import Merlin.GraphStateScalarSound

namespace Merlin.GraphState.Pipeline.Inline
set_option autoImplicit false
open Scalar GraphLoopsUnroll

/-- No body operation is invoked in a counted empty source loop. This includes
arbitrary partial/effectful bodies and retains the initial carry and world. -/
theorem empty_no_body {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (lower upper step : Nat) (positive : 0 < step)
    (counted : CountCorrect lower upper step 0)
    (values : List Value) (world : World) (size : values.length = arity) :
    loopEval arity body (indexIterations lower upper step) values world = .ok (values,world) := by
  rw [indexIterations_count positive counted]
  simp only [progression,List.map_nil,loopEval,size,↓reduceIte]

/-- Counted source execution, INCLUDING its continuation, can be reduced over
the exact lane IV progression. Instantiation is bounded by the selected inline
factor; no unbounded trip-sized trace or success premise is introduced. -/
theorem counted_continuation {Value World Error : Type} (arity : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (next : List Value → World → Outcome Value World Error)
    (lower upper step count : Nat) (positive : 0 < step)
    (counted : CountCorrect lower upper step count)
    (values : List Value) (world : World) :
    (loopEval arity body (indexIterations lower upper step) values world).bind
      (fun result => next result.1 result.2) =
    (loopEval arity body ((progression lower step count).map Int.ofNat) values world).bind
      (fun result => next result.1 result.2) := by
  rw [indexIterations_count positive counted]

/-- A failing inline lane suppresses all remaining lanes and the post-loop
observer. The failure includes the terminal world, not only an error code. -/
theorem failed_continuation {Value World Error : Type}
    (failure : Fault Error × World)
    (next : List Value → World → Outcome Value World Error) :
    (Except.error failure : Outcome Value World Error).bind
      (fun result => next result.1 result.2) = .error failure := rfl

end Merlin.GraphState.Pipeline.Inline
