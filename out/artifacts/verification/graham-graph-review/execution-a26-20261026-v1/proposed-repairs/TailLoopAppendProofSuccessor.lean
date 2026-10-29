import Merlin.GraphStatePipelineLoops
namespace Merlin.GraphState.Pipeline.Tail
set_option autoImplicit false
open Scalar GraphLoopsUnroll
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


end Merlin.GraphState.Pipeline.Tail
#check @Merlin.GraphState.Pipeline.Tail.loop_append
#print axioms Merlin.GraphState.Pipeline.Tail.loop_append
