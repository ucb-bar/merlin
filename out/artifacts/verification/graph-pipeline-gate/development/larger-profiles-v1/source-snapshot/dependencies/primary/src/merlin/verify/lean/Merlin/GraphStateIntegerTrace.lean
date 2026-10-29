/- Every actually reached operand's RUNTIME representation, separately from
the old slot/initialization theorem. Errors stop the trace; success is not assumed.
This uses the unchanged Scalar evaluator, including nested loops/captures/carries. -/
import Merlin.GraphStateIntegerRepresentation

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar

def loopReached {World Error : Type} (rep : Ty → Int → Prop) (types : List Ty)
    (safe : Int → List Int → World → Prop) (body : Int → List Int → World → Outcome Int World Error) :
    List Int → List Int → World → Prop
  | [],values,_ => Values rep types values
  | index::later,values,world => Values rep types values ∧ safe index values world ∧
    match body index values world with
    | .error _ => True
    | .ok (next,final) => loopReached rep types safe body later next final

def Reached {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (rep : Ty → Int → Prop) :
    Program → List Int → World → Prop
  | .ret _,_,_ => True
  | .pure op arguments next,environment,world =>
    Values rep op.inputs (reads model.default environment arguments) ∧
    rep op.output (model.primitive op (reads model.default environment arguments)) ∧
    Reached model rep next (model.primitive op (reads model.default environment arguments)::environment) world
  | .opaque identity arguments _ next,environment,world =>
    Values rep (signature registry identity).inputs (reads model.default environment arguments) ∧
    match opaqueStep model identity (reads model.default environment arguments) world with
    | .error _ => True
    | .ok (values,final) =>
      Values rep (signature registry identity).outputs values ∧ Reached model rep next (values++environment) final
  | .loop lower upper stride initial carries body next,environment,world =>
    Values rep [.index,.index,.index] (reads model.default environment [lower,upper,stride]) ∧
    loopReached rep carries
      (fun index carry current => Reached model rep body (model.index index::carry++environment) current)
      (fun index carry current => evalProgram model body (model.index index::carry++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world ∧
    match loopEval carries.length
      (fun index carry current => evalProgram model body (model.index index::carry++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world with
    | .error _ => True
    | .ok (values,final) => Values rep carries values ∧ Reached model rep next (values++environment) final

theorem loop_reached {World Error : Type} {rep : Ty → Int → Prop} (types : List Ty)
    (safe : Int → List Int → World → Prop) (body : Int → List Int → World → Outcome Int World Error)
    (bodySafe : ∀ index values world, Values rep types values → safe index values world)
    (bodyTyped : ∀ index values world, Values rep types values → ResultTyped rep types (body index values world))
    (iterations : List Int) (values : List Int) (world : World) (typed : Values rep types values) :
    loopReached rep types safe body iterations values world := by
  induction iterations generalizing values world with
  | nil => exact typed
  | cons index later ih =>
    refine ⟨typed,bodySafe index values world typed,?_⟩
    have shape := bodyTyped index values world typed
    cases outcome : body index values world with
    | error failure => trivial
    | ok result => exact ih result.1 result.2 (by simpa only [outcome,ResultTyped] using shape)

theorem typed_reached {World Error : Type} {registry : List Signature} {rep : Ty → Int → Prop}
    (model : Model Int World Error registry) (laws : LocalTyping model rep)
    {types results : List Ty} {program : Program} (typed : Typed registry types results program)
    (values : List Int) (world : World) (entry : Values rep types values) : Reached model rep program values world := by
  induction typed generalizing values world with
  | ret operands => trivial
  | pure eligible operands later ih =>
    have inputs := entry.reads operands model.default
    have output := laws.primitive _ _ eligible inputs
    exact ⟨inputs,output,ih _ world (.cons output entry)⟩
  | @effect types results outputs identity arguments selected next known equal operands later ih =>
    have inputs : Values rep (signature registry identity).inputs (reads model.default values arguments) := by
      simpa only [signature,known,Option.getD_some] using entry.reads operands model.default
    refine ⟨inputs,?_⟩
    cases step : model.effect identity (reads model.default values arguments) world with
    | error failure => simp only [opaqueStep,step]
    | ok result =>
      simp only [opaqueStep,step]
      have outputsTyped := laws.effect _ _ _ _ _ inputs step
      refine ⟨outputsTyped,?_⟩
      apply ih
      have same : (signature registry identity).outputs = outputs := by simp [signature,known,equal]
      rw [same] at outputsTyped
      exact outputsTyped.append entry
  | @loop types results carries lower upper stride initial body next bounds operands inside later ihBody ihNext =>
    refine ⟨entry.reads bounds model.default,?_,?_⟩
    · apply loop_reached
      · intro index carry current carryTyped
        exact ihBody _ current (.cons (laws.index index) (carryTyped.append entry))
      · intro index carry current carryTyped
        exact typed_result model laws inside _ current (.cons (laws.index index) (carryTyped.append entry))
      · exact entry.reads operands model.default
    · have shape := loop_result_typed carries
        (fun index carry current => evalProgram model body (model.index index::carry++values) current)
        (fun index carry current carryTyped => typed_result model laws inside _ current
          (.cons (laws.index index) (carryTyped.append entry)))
        (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
          (values[stride]?.getD model.default)) (reads model.default values initial) world
        (entry.reads operands model.default)
      cases outcome : loopEval carries.length
          (fun index carry current => evalProgram model body (model.index index::carry++values) current)
          (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
            (values[stride]?.getD model.default)) (reads model.default values initial) world with
      | error failure => trivial
      | ok result =>
        have carryTyped : Values rep carries result.1 := by simpa only [outcome,ResultTyped] using shape
        exact ⟨carryTyped,ihNext _ result.2 (carryTyped.append entry)⟩

#print axioms typed_reached
end Merlin.GraphState.Integer
