/- Normalization is operationally sound, including partial opaque steps,
zero-trip loops, exact carry/result arity and complete terminal observations. -/
import Merlin.GraphStateScalarNormalize

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

theorem opaqueStep_arity {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) {identity : Nat} {arguments : List Value}
    {world final : World} {values : List Value}
    (success : opaqueStep model identity arguments world = .ok (values,final)) :
    values.length = (signature registry identity).outputs.length := by
  cases actual : model.effect identity arguments world with
  | error failure => simp [opaqueStep,actual] at success
  | ok result =>
    have same : result = (values,final) := by simpa only [opaqueStep,actual,Except.ok.injEq] using success
    subst result
    exact model.arity identity arguments world values final actual

theorem loopEval_arity {Value World Error : Type} (count : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (iterations : List Int) (values : List Value) (world : World) {result : List Value × World}
    (success : loopEval count body iterations values world = .ok result) : result.1.length = count := by
  induction iterations generalizing values world with
  | nil =>
    simp only [loopEval] at success
    split at success
    · cases success; assumption
    · cases success
  | cons index rest ih =>
    simp only [loopEval] at success
    split at success
    · cases step : body index values world with
      | error failure => simp only [step,Except.bind] at success; cases success
      | ok next =>
        simp only [step,Except.bind] at success
        exact ih next.1 next.2 success
    · cases success

theorem loopEval_congr {Value World Error : Type} (count : Nat)
    (first second : Int → List Value → World → Outcome Value World Error)
    (same : ∀ index values world, values.length = count → first index values world = second index values world)
    (iterations : List Int) (values : List Value) (world : World) :
    loopEval count first iterations values world = loopEval count second iterations values world := by
  induction iterations generalizing values world with
  | nil => rfl
  | cons index rest ih =>
    simp only [loopEval]
    split
    · next size =>
      rw [same index values world size]
      cases step : second index values world with
      | error failure => rfl
      | ok result => simpa only [Except.bind] using ih result.1 result.2
    · rfl

theorem normalize_sound {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) (program : Program) :
    ∀ symbols code, normalize registry symbols program = some code →
      ∀ environment world,
      evalProgram model program (symbols.map (evalExpr model.default model.primitive environment)) world =
        evalCode model code environment world := by
  induction program with
  | ret values =>
    intro symbols code checked environment world
    cases operands : lookupAll symbols values with
    | none => simp [normalize,operands] at checked
    | some expressions =>
      have equal : Code.ret expressions = code := by simpa [normalize,operands] using checked
      subst code
      simp only [evalProgram,evalCode,lookupAll_sound model.default model.primitive environment operands]
  | pure primitive arguments next ih =>
    intro symbols code checked environment world
    cases operands : lookupAll symbols arguments with
    | none => simp [normalize,operands] at checked
    | some expressions =>
      cases made : makeExpr primitive expressions with
      | none => simp [normalize,operands,made] at checked
      | some expression =>
        have remainder : normalize registry (expression::symbols) next = some code := by
          simpa [normalize,operands,made] using checked
        have nextEqual := ih (expression::symbols) code remainder environment world
        simpa only [evalProgram,List.map_cons,makeExpr_sound model.default model.primitive environment made,
          lookupAll_sound model.default model.primitive environment operands] using nextEqual
  | «opaque» identity arguments results next ih =>
    intro symbols code checked environment world
    by_cases valid : identity < registry.length ∧ results = (signature registry identity).outputs
    · cases operands : lookupAll symbols arguments with
      | none => simp [normalize,valid,operands] at checked
      | some expressions =>
        cases later : normalize registry (lift results.length symbols) next with
        | none => simp only [normalize,ite_eq_left valid,operands,later,Option.bind_eq_bind,Option.bind_some,Option.bind_none,Option.pure_def] at checked; cases checked
        | some continuation =>
          have equal : Code.opaque identity expressions results continuation = code := by
            simpa only [normalize,ite_eq_left valid,operands,later,Option.bind_eq_bind,Option.bind_some,Option.pure_def,Option.some.injEq] using checked
          subst code
          have argumentsEqual := lookupAll_sound model.default model.primitive environment operands
          simp only [evalProgram,evalCode,argumentsEqual]
          cases step : opaqueStep model identity (expressions.map (evalExpr model.default model.primitive environment)) world with
          | error failure => rfl
          | ok result =>
            have size : result.1.length = results.length := (opaqueStep_arity model step).trans (congrArg List.length valid.2).symm
            have binding := eval_lift model.default model.primitive result.1 environment symbols
            rw [size] at binding
            have same := ih (lift results.length symbols) continuation later (result.1++environment) result.2
            simpa only [Except.bind,binding] using same
    · simp [normalize,valid] at checked
  | loop lower upper stride initial carries body next ihBody ihNext =>
    intro symbols code checked environment world
    cases lbFound : symbols[lower]? with
    | none => simp [normalize,lbFound] at checked
    | some lb =>
      cases ubFound : symbols[upper]? with
      | none => simp [normalize,lbFound,ubFound] at checked
      | some ub =>
        cases stepFound : symbols[stride]? with
        | none => simp [normalize,lbFound,ubFound,stepFound] at checked
        | some strideExpr =>
          cases valuesFound : lookupAll symbols initial with
          | none => simp [normalize,lbFound,ubFound,stepFound,valuesFound] at checked
          | some initialExprs =>
            cases inside : normalize registry (lift (carries.length+1) symbols) body with
            | none => simp [normalize,lbFound,ubFound,stepFound,valuesFound,inside] at checked
            | some inner =>
              cases outside : normalize registry (lift carries.length symbols) next with
              | none => simp [normalize,lbFound,ubFound,stepFound,valuesFound,inside,outside] at checked
              | some continuation =>
                have equal : Code.loop lb ub strideExpr initialExprs carries inner continuation = code := by
                  simpa [normalize,lbFound,ubFound,stepFound,valuesFound,inside,outside] using checked
                subst code
                have iterationEqual : ∀ iterations values current,
                    loopEval carries.length
                      (fun index carry state => evalProgram model body
                        (model.index index::carry++symbols.map (evalExpr model.default model.primitive environment)) state)
                      iterations values current =
                    loopEval carries.length
                      (fun index carry state => evalCode model inner (model.index index::carry++environment) state)
                      iterations values current := by
                  intro iterations values current
                  apply loopEval_congr
                  intro index carry state size
                  have binding := eval_lift model.default model.primitive (model.index index::carry) environment symbols
                  have count : (model.index index::carry).length = carries.length+1 := by simp [size]
                  rw [count] at binding
                  simp only [List.cons_append] at binding
                  have same := ihBody (lift (carries.length+1) symbols) inner inside (model.index index::carry++environment) state
                  simp only [List.cons_append] at same ⊢
                  rw [binding] at same
                  exact same
                have initialEqual := lookupAll_sound model.default model.primitive environment valuesFound
                simp only [evalProgram,evalCode,List.getElem?_map,lbFound,ubFound,stepFound,Option.map_some,
                  Option.getD_some,initialEqual,iterationEqual]
                cases outcome : loopEval carries.length
                    (fun index carry state => evalCode model inner (model.index index::carry++environment) state)
                    (model.iterations (evalExpr model.default model.primitive environment lb)
                      (evalExpr model.default model.primitive environment ub)
                      (evalExpr model.default model.primitive environment strideExpr))
                    (initialExprs.map (evalExpr model.default model.primitive environment)) world with
                | error failure => rfl
                | ok result =>
                  have size := loopEval_arity carries.length _ _ _ _ outcome
                  have binding := eval_lift model.default model.primitive result.1 environment symbols
                  rw [size] at binding
                  have same := ihNext (lift carries.length symbols) continuation outside (result.1++environment) result.2
                  simpa only [Except.bind,binding] using same

#print axioms normalize_sound
#print axioms loopEval_congr
end Merlin.GraphState.Scalar
