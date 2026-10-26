/- Expose the actual normalized syntax without conflating the two evaluators.
The source blueprint's candidate mode is reduced to an explicit ordinary Code
with cast aliases resolved and dimensions replaced by their checked constants. -/
import Merlin.GraphLoopsShapeEpochViews

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def rename (aliases : Aliases) (binding : Binding) : Binding :=
  { binding with name := resolve aliases binding.name }

def normalized (aliases : Aliases) : Code → Code
  | .returned values => .returned (values.map (rename aliases))
  | .cast origin result _ _ rest => normalized ((result.name, resolve aliases origin.name) :: aliases) rest
  | .dimension _ _ result _ _ value rest =>
      .constant result value (normalized ((result.name, result.name) :: aliases) rest)
  | .constant result value rest =>
      .constant result value (normalized ((result.name, result.name) :: aliases) rest)
  | .action key operands results rest =>
      .action key (operands.map (rename aliases)) results (normalized (extendAliases aliases results) rest)
  | .region key operands parameters results body rest =>
      .region key (operands.map (rename aliases)) parameters results
        (normalized (extendAliases aliases parameters) body) (normalized (extendAliases aliases results) rest)

theorem renamed_arguments {Value : Type} (aliases other : Aliases) (env : Env Value) (bindings : List Binding) :
    readArguments false other env (bindings.map (rename aliases)) = readArguments true aliases env bindings := by
  simp only [readArguments, List.map_map, rename, Bool.false_eq_true, ↓reduceIte, Function.comp_def]

/-- This is a syntax-derived equality for every shared primitive/region meaning.
There is no hypothesis of whole-program agreement and no new primitive law. -/
theorem normalized_eval {Value World : Type} (meaning : Interpretation Value World) (code : Code)
    (aliases other : Aliases) (env : Env Value) (before after : World) (outputs : List (Runtime Value)) :
    Eval meaning true aliases code env before outputs after ↔
      Eval meaning false other (normalized aliases code) env before outputs after := by
  induction code generalizing aliases other env before after outputs with
  | returned values => simp only [Eval, normalized, renamed_arguments]
  | cast origin result source target rest induction =>
    simpa only [Eval, normalized, ↓reduceIte] using
      induction ((result.name, resolve aliases origin.name) :: aliases) other env before after outputs
  | dimension view axis result type axisIndex value rest induction =>
    simpa only [Eval, normalized, ↓reduceIte] using
      induction ((result.name, result.name) :: aliases) ((result.name, result.name) :: other)
        (put env result.name (.index value)) before after outputs
  | constant result value rest induction =>
    exact induction _ _ _ _ _ _
  | action key operands results rest induction =>
    simp only [Eval, normalized, renamed_arguments]
    constructor
    · rintro ⟨values, middle, step, typed, suffix⟩
      exact ⟨values, middle, step, typed, (induction _ _ _ _ _ _).mp suffix⟩
    · rintro ⟨values, middle, step, typed, suffix⟩
      exact ⟨values, middle, step, typed, (induction _ _ _ _ _ _).mpr suffix⟩
  | region key operands parameters results body rest bodyInduction restInduction =>
    have bodyEqual :
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
          Eval meaning true (extendAliases aliases parameters) body (assign env parameters inputs) initial returned final) =
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
          Eval meaning false (extendAliases other parameters) (normalized (extendAliases aliases parameters) body)
            (assign env parameters inputs) initial returned final) := by
      funext inputs initial returned final
      apply propext
      exact and_congr_right (fun _ => bodyInduction _ _ _ _ _ _)
    simp only [Eval, normalized, renamed_arguments]
    rw [bodyEqual]
    constructor
    · rintro ⟨values, middle, step, typed, suffix⟩
      exact ⟨values, middle, step, typed, (restInduction _ _ _ _ _ _).mp suffix⟩
    · rintro ⟨values, middle, step, typed, suffix⟩
      exact ⟨values, middle, step, typed, (restInduction _ _ _ _ _ _).mpr suffix⟩

def normalizedFunction (function : Function) : Function :=
  { function with code := normalized (extendAliases [] function.parameters) function.code }

def normalizedModule (module : Module) : Module :=
  ⟨module.functions.map normalizedFunction, module.calls⟩

theorem normalized_function {Value World : Type} (function : Function) (meaning : Interpretation Value World) :
    function.invoke meaning true = (normalizedFunction function).invoke meaning false := by
  funext inputs before outputs after
  apply propext
  exact and_congr_right (fun _ => normalized_eval meaning function.code _ _ _ _ _ _)

theorem normalized_find (functions : List Function) (name : String) :
    (functions.map normalizedFunction).find? (fun function => function.name == name) =
      (functions.find? (fun function => function.name == name)).map normalizedFunction := by
  induction functions with
  | nil => rfl
  | cons function rest induction =>
    simp only [List.map_cons, List.find?_cons, normalizedFunction]
    by_cases equal : function.name == name
    · simp [equal, normalizedFunction]
    · simpa only [equal, Bool.false_eq_true, ↓reduceIte] using induction

theorem normalized_functionAt (module : Module) (name : String) :
    functionAt (normalizedModule module) name = (functionAt module name).map normalizedFunction :=
  normalized_find module.functions name

theorem normalized_withCalls {Value World : Type} (meaning : Interpretation Value World) (module : Module)
    (calls : String → RegionMeaning Value World) :
    withCalls meaning (normalizedModule module) calls = withCalls meaning module calls := by
  unfold withCalls
  congr 1
  funext key inputs before outputs after
  change (match callTarget module.calls key with
    | some name => if (functionAt (normalizedModule module) name).isSome then
        calls name inputs before outputs after else meaning.action key inputs before outputs after
    | none => meaning.action key inputs before outputs after) = _
  cases target : callTarget module.calls key with
  | none => rfl
  | some name =>
    simp only [normalized_functionAt]
    cases found : functionAt module name <;> rfl

theorem normalized_invoke {Value World : Type} (meaning : Interpretation Value World) (module : Module) :
    ∀ depth, invoke meaning module true depth = invoke meaning (normalizedModule module) false depth := by
  intro depth
  induction depth with
  | zero => rfl
  | succ depth induction =>
    funext name
    simp only [invoke, normalized_functionAt]
    cases found : functionAt module name with
    | none => rfl
    | some function =>
      simp only [Option.map_some]
      rw [induction, normalized_withCalls]
      exact normalized_function function _

theorem normalized_complete {Value World : Type} (meaning : Interpretation Value World) (module : Module)
    (name : String) (inputs : List (Runtime Value)) (before after : World) (outputs : List (Runtime Value)) :
    complete meaning module true name inputs before outputs after ↔
      complete meaning (normalizedModule module) false name inputs before outputs after := by
  unfold complete
  apply exists_congr
  intro depth
  rw [normalized_invoke]

end Merlin.GraphLoopsShape.Epoch
