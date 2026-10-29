/- Complete contextual correspondence on the EXISTING Scalar.Code evaluator.
The relation is a checked local rewrite sequence: it fixes every retained call,
loop and observation while proving every changed expression by local identities.
No whole-program equality or successful execution is an input premise. -/
import Merlin.GraphStateIntegerExpressions

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar

inductive Relation (registry : List Signature) : List Ty → List Ty → Code → Code → Prop where
  | ret {environment results : List Ty} {first second : List Expr}
      (typed : ExprsTyped environment first results) (same : Sames environment first second) :
      Relation registry environment results (.ret first) (.ret second)
  | effect {environment results outputs : List Ty} {identity : Nat} {first second : List Expr}
      {next later : Code} (known : identity < registry.length)
      (outputTypes : outputs = (signature registry identity).outputs)
      (typed : ExprsTyped environment first (signature registry identity).inputs)
      (same : Sames environment first second)
      (rest : Relation registry (outputs++environment) results next later) :
      Relation registry environment results (.opaque identity first outputs next) (.opaque identity second outputs later)
  | loop {environment results carries : List Ty} {lower upper stride lb ub step : Expr}
      {first second : List Expr} {body other next later : Code}
      (lowerTyped : ExprTyped environment lower .index) (upperTyped : ExprTyped environment upper .index)
      (strideTyped : ExprTyped environment stride .index)
      (initialTyped : ExprsTyped environment first carries)
      (lowerSame : Same environment lower lb) (upperSame : Same environment upper ub)
      (strideSame : Same environment stride step) (initialSame : Sames environment first second)
      (inside : Relation registry (.index::carries++environment) carries body other)
      (rest : Relation registry (carries++environment) results next later) :
      Relation registry environment results (.loop lower upper stride first carries body next)
        (.loop lb ub step second carries other later)

theorem loops_sound {World Error : Type} (types : List Ty)
    (first second : Int → List Int → World → Outcome Int World Error)
    (body : ∀ index values world, Values Canonical types values →
      ResultTyped Canonical types (first index values world) ∧ first index values world = second index values world)
    (iterations : List Int) (values : List Int) (world : World) (typed : Values Canonical types values) :
    ResultTyped Canonical types (loopEval types.length first iterations values world) ∧
      loopEval types.length first iterations values world = loopEval types.length second iterations values world := by
  refine ⟨loop_result_typed types first (fun index values world h => (body index values world h).1)
    iterations values world typed,?_⟩
  induction iterations generalizing values world with
  | nil => rfl
  | cons index rest ih =>
    simp only [loopEval,typed.length,↓reduceIte]
    have current := body index values world typed
    rw [← current.2]
    cases step : first index values world with
    | error failure => rfl
    | ok result =>
      exact ih result.1 result.2 (by simpa only [step,ResultTyped] using current.1)

theorem Relation.sound {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (exact : ExactModel model)
    (laws : LocalTyping model Canonical) {environment results : List Ty} {first second : Code}
    (related : Relation registry environment results first second)
    (values : List Int) (world : World) (entry : Values Canonical environment values) :
    ResultTyped Canonical results (evalCode model first values world) ∧
      evalCode model first values world = evalCode model second values world := by
  induction related generalizing values world with
  | ret typed same =>
    constructor
    · simpa only [evalCode,ResultTyped,exact.1,exact.2] using typed.canonical values entry
    · simp only [evalCode,exact.1,exact.2,same values entry]
  | @effect environment results outputs identity first second next later known outputTypes typed same rest ih =>
    have argumentsTyped : Values Canonical (signature registry identity).inputs
        (first.map (evalExpr model.default model.primitive values)) := by
      simpa only [exact.1,exact.2] using typed.canonical values entry
    have argumentsSame : first.map (evalExpr model.default model.primitive values) =
        second.map (evalExpr model.default model.primitive values) := by
      simpa only [exact.1,exact.2] using same values entry
    simp only [evalCode]
    rw [← argumentsSame]
    cases step : model.effect identity (first.map (evalExpr model.default model.primitive values)) world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      simp only [opaqueStep,step,Except.bind,ResultTyped]
      constructor <;> trivial
    | ok result =>
      have resultTyped := laws.effect _ _ _ _ _ argumentsTyped step
      rw [← outputTypes] at resultTyped
      simpa only [opaqueStep,step,Except.bind] using ih _ result.2 (resultTyped.append entry)
  | @loop environment results carries lower upper stride lb ub step first second body other next later
      lowerTyped upperTyped strideTyped initialTyped lowerSame upperSame strideSame initialSame inside rest ihBody ihNext =>
    have initial : Values Canonical carries (first.map (evalExpr model.default model.primitive values)) := by
      simpa only [exact.1,exact.2] using initialTyped.canonical values entry
    have initialEqual : first.map (evalExpr model.default model.primitive values) =
        second.map (evalExpr model.default model.primitive values) := by
      simpa only [exact.1,exact.2] using initialSame values entry
    have loop := loops_sound carries
      (fun index carry current => evalCode model body (model.index index::carry++values) current)
      (fun index carry current => evalCode model other (model.index index::carry++values) current)
      (fun index carry current carryTyped => ihBody _ current
        (.cons (laws.index index) (carryTyped.append entry)))
      (model.iterations (evalExpr model.default model.primitive values lower)
        (evalExpr model.default model.primitive values upper) (evalExpr model.default model.primitive values stride))
      (first.map (evalExpr model.default model.primitive values)) world initial
    simp only [evalCode]
    rw [← lowerSame.model model exact values entry,← upperSame.model model exact values entry,
      ← strideSame.model model exact values entry,← initialEqual,← loop.2]
    cases outcome : loopEval carries.length
        (fun index carry current => evalCode model body (model.index index::carry++values) current)
        (model.iterations (evalExpr model.default model.primitive values lower)
          (evalExpr model.default model.primitive values upper) (evalExpr model.default model.primitive values stride))
        (first.map (evalExpr model.default model.primitive values)) world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      cases error with
      | external error => exact ⟨True.intro,rfl⟩
      | arity =>
        have impossible : False := by simpa only [outcome,ResultTyped] using loop.1
        exact impossible.elim
    | ok result =>
      have resultTyped : Values Canonical carries result.1 := by
        simpa only [outcome,ResultTyped] using loop.1
      exact ihNext _ result.2 (resultTyped.append entry)

/-- The submitted complete programs are independently normalized by the old
proved SSA normalizer. The local relation supplies the new integer proof; it
does not compare independently numbered private SSA environments. -/
theorem programs_sound {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (exact : ExactModel model) (laws : LocalTyping model Canonical)
    {inputs outputs : List Ty} {before after : Program} {first second : Code}
    (left : normalize registry (vars inputs.length) before = some first)
    (right : normalize registry (vars inputs.length) after = some second)
    (related : Relation registry inputs outputs first second)
    (values : List Int) (world : World) (entry : Values Canonical inputs values) :
    ResultTyped Canonical outputs (evalProgram model before values world) ∧
      evalProgram model before values world = evalProgram model after values world := by
  have binding := eval_vars model.default model.primitive values []
  simp only [List.append_nil] at binding
  rw [entry.length] at binding
  have source := normalize_sound model before _ _ left values world
  have target := normalize_sound model after _ _ right values world
  rw [binding] at source target
  have proof := related.sound model exact laws values world entry
  rw [source,target]
  exact proof

#print axioms Relation.sound
#print axioms programs_sound
end Merlin.GraphState.Integer
