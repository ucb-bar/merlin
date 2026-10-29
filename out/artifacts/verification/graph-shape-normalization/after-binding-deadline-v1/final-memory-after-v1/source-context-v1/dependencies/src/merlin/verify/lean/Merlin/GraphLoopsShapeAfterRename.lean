/- Candidate: bijective SSA renaming of an independently decoded actual module.
UNCOMPILED while the selected native runtime is held. No old semantic module is
changed. Name swaps preserve complete typed evaluation, including region scopes
and finite internal-call derivations; they do not rename operation meanings. -/
import Merlin.GraphLoopsShapeEpochNormalize

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.After

def swapName (left right name : Nat) : Nat :=
  if name = left then right else if name = right then left else name

theorem swap_twice (left right name : Nat) :
    swapName left right (swapName left right name) = name := by
  by_cases same : left = right
  · subst right
    by_cases hit : name = left <;> simp_all [swapName]
  · by_cases first : name = left
    · subst name
      simp [swapName, same, Ne.symm same]
    · by_cases second : name = right
      · subst name
        simp [swapName, same, Ne.symm same]
      · simp [swapName, first, second]

theorem swap_injective (left right : Nat) : Function.Injective (swapName left right) := by
  intro first second equal
  have twice := congrArg (swapName left right) equal
  simpa only [swap_twice] using twice

def swapBinding (left right : Nat) (binding : Binding) : Binding :=
  { binding with name := swapName left right binding.name }

theorem binding_twice (left right : Nat) (binding : Binding) :
    swapBinding left right (swapBinding left right binding) = binding := by
  cases binding
  simp only [swapBinding, swap_twice]

def swapEnv {Value : Type} (left right : Nat) (env : Env Value) : Env Value :=
  fun name => env (swapName left right name)

theorem put_swapped {Value : Type} (left right name : Nat) (env : Env Value) (value : Runtime Value) :
    swapEnv left right (put env name value) =
      put (swapEnv left right env) (swapName left right name) value := by
  funext other
  have equal : swapName left right other = name ↔ other = swapName left right name := by
    constructor
    · intro same
      have twice := congrArg (swapName left right) same
      simpa only [swap_twice] using twice
    · intro same
      rw [same, swap_twice]
  simp only [swapEnv, put, equal]

theorem assign_swapped {Value : Type} (left right : Nat) (bindings : List Binding)
    (values : List (Runtime Value)) (env : Env Value) :
    swapEnv left right (assign env bindings values) =
      assign (swapEnv left right env) (bindings.map (swapBinding left right)) values := by
  induction bindings generalizing values env with
  | nil => rfl
  | cons binding rest induction =>
    cases values with
    | nil => rfl
    | cons value values =>
      simp only [assign, List.map_cons]
      rw [induction, put_swapped]
      rfl

theorem values_swapped {Value : Type} (left right : Nat) {bindings : List Binding}
    {values : List (Runtime Value)} (typed : ValuesHold bindings values) :
    ValuesHold (bindings.map (swapBinding left right)) values := by
  induction typed with
  | nil => exact .nil
  | cons holds typed induction => exact .cons holds induction

theorem values_swapped_iff {Value : Type} (left right : Nat) (bindings : List Binding)
    (values : List (Runtime Value)) :
    ValuesHold (bindings.map (swapBinding left right)) values ↔ ValuesHold bindings values := by
  constructor
  · intro typed
    have twice := values_swapped left right typed
    simpa [List.map_map, Function.comp_def, binding_twice] using twice
  · exact values_swapped left right

theorem arguments_swapped {Value : Type} (left right : Nat) (aliases other : Aliases)
    (bindings : List Binding) (env : Env Value) :
    readArguments false other (swapEnv left right env) (bindings.map (swapBinding left right)) =
      readArguments false aliases env bindings := by
  simp only [readArguments, Bool.false_eq_true, ↓reduceIte, List.map_map,
    Function.comp_def, swapBinding, swapEnv, swap_twice]

def swapCode (left right : Nat) : Code → Code
  | .returned values => .returned (values.map (swapBinding left right))
  | .cast origin result source target rest =>
      .cast (swapBinding left right origin) (swapBinding left right result) source target (swapCode left right rest)
  | .dimension view axis result type axisIndex value rest =>
      .dimension (swapBinding left right view) (swapBinding left right axis) (swapBinding left right result)
        type axisIndex value (swapCode left right rest)
  | .constant result value rest => .constant (swapBinding left right result) value (swapCode left right rest)
  | .action key operands results rest =>
      .action key (operands.map (swapBinding left right)) (results.map (swapBinding left right)) (swapCode left right rest)
  | .region key operands parameters results body rest =>
      .region key (operands.map (swapBinding left right)) (parameters.map (swapBinding left right))
        (results.map (swapBinding left right)) (swapCode left right body) (swapCode left right rest)

/-- Both sides are ordinary-code evaluation. No candidate-mode equality or
opaque whole-program correspondence is assumed. -/
theorem eval_swapped {Value World : Type} (left right : Nat) (meaning : Interpretation Value World)
    (code : Code) (aliases other : Aliases) (env : Env Value) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases code env before outputs after ↔
      Eval meaning false other (swapCode left right code) (swapEnv left right env) before outputs after := by
  induction code generalizing aliases other env before outputs after with
  | returned values => simp only [Eval, swapCode, arguments_swapped]
  | cast origin result source target rest induction =>
    simp only [Eval, swapCode, Bool.false_eq_true, ↓reduceIte, swapBinding, swapEnv, swap_twice]
    constructor
    · rintro ⟨value, found, suffix⟩
      refine ⟨value, found, ?_⟩
      have next := (induction _ _ _ _ _ _).mp suffix
      simpa only [put_swapped] using next
    · rintro ⟨value, found, suffix⟩
      refine ⟨value, found, ?_⟩
      apply (induction _ _ _ _ _ _).mpr
      simpa only [put_swapped] using suffix
  | dimension view axis result type axisIndex value rest induction =>
    simp only [Eval, swapCode, Bool.false_eq_true, ↓reduceIte, swapBinding, swapEnv, swap_twice]
    constructor
    · rintro ⟨actual, found, suffix⟩
      refine ⟨actual, found, ?_⟩
      have next := (induction _ _ _ _ _ _).mp suffix
      simpa only [put_swapped] using next
    · rintro ⟨actual, found, suffix⟩
      refine ⟨actual, found, ?_⟩
      apply (induction _ _ _ _ _ _).mpr
      simpa only [put_swapped] using suffix
  | constant result value rest induction =>
    simpa only [Eval, swapCode, put_swapped] using
      induction ((result.name, result.name) :: aliases)
        (((swapBinding left right result).name, (swapBinding left right result).name) :: other)
        (put env result.name (.index value)) before outputs after
  | action key operands results rest induction =>
    simp only [Eval, swapCode, arguments_swapped, values_swapped_iff]
    constructor
    · rintro ⟨values, middle, step, typed, suffix⟩
      refine ⟨values, middle, step, typed, ?_⟩
      simpa only [assign_swapped] using (induction _ _ _ _ _ _).mp suffix
    · rintro ⟨values, middle, step, typed, suffix⟩
      refine ⟨values, middle, step, typed, ?_⟩
      apply (induction _ _ _ _ _ _).mpr
      simpa only [assign_swapped] using suffix
  | region key operands parameters results body rest bodyInduction restInduction =>
    have bodyEqual :
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
          Eval meaning false (extendAliases aliases parameters) body (assign env parameters inputs) initial returned final) =
        (fun inputs initial returned final => ValuesHold (parameters.map (swapBinding left right)) inputs ∧
          Eval meaning false (extendAliases other (parameters.map (swapBinding left right))) (swapCode left right body)
            (assign (swapEnv left right env) (parameters.map (swapBinding left right)) inputs) initial returned final) := by
      funext inputs initial returned final
      apply propext
      rw [values_swapped_iff]
      apply and_congr_right
      intro _
      simpa only [assign_swapped] using bodyInduction _ _ (assign env parameters inputs) initial returned final
    simp only [Eval, swapCode, arguments_swapped, values_swapped_iff]
    rw [bodyEqual]
    constructor
    · rintro ⟨values, middle, step, typed, suffix⟩
      refine ⟨values, middle, step, typed, ?_⟩
      simpa only [assign_swapped] using (restInduction _ _ _ _ _ _).mp suffix
    · rintro ⟨values, middle, step, typed, suffix⟩
      refine ⟨values, middle, step, typed, ?_⟩
      apply (restInduction _ _ _ _ _ _).mpr
      simpa only [assign_swapped] using suffix

def swapFunction (left right : Nat) (function : Function) : Function :=
  { function with parameters := function.parameters.map (swapBinding left right), code := swapCode left right function.code }

def swapModule (left right : Nat) (module : Module) : Module :=
  ⟨module.functions.map (swapFunction left right), module.calls⟩

theorem function_swapped {Value World : Type} (left right : Nat) (function : Function)
    (meaning : Interpretation Value World) :
    function.invoke meaning false = (swapFunction left right function).invoke meaning false := by
  funext inputs before outputs after
  apply propext
  simp only [Function.invoke, swapFunction, values_swapped_iff]
  apply and_congr_right
  intro _
  have related := eval_swapped left right meaning function.code _ _
    (assign emptyEnv function.parameters inputs) before outputs after
  simpa only [assign_swapped, swapEnv, emptyEnv] using related

theorem functionAt_swapped (left right : Nat) (module : Module) (name : String) :
    functionAt (swapModule left right module) name = (functionAt module name).map (swapFunction left right) := by
  unfold functionAt swapModule
  induction module.functions with
  | nil => rfl
  | cons function rest induction =>
    simp only [List.map_cons, List.find?_cons, swapFunction]
    by_cases equal : function.name == name
    · simp [equal, swapFunction]
    · simpa only [equal, Bool.false_eq_true, ↓reduceIte] using induction

theorem withCalls_swapped {Value World : Type} (left right : Nat) (meaning : Interpretation Value World)
    (module : Module) (calls : String → RegionMeaning Value World) :
    withCalls meaning (swapModule left right module) calls = withCalls meaning module calls := by
  unfold withCalls
  congr 1
  funext key inputs before outputs after
  change (match callTarget module.calls key with
    | some name => if (functionAt (swapModule left right module) name).isSome then
        calls name inputs before outputs after else meaning.action key inputs before outputs after
    | none => meaning.action key inputs before outputs after) = _
  cases target : callTarget module.calls key with
  | none => rfl
  | some name =>
    simp only [functionAt_swapped]
    cases found : functionAt module name <;> rfl

theorem invoke_swapped {Value World : Type} (left right : Nat) (meaning : Interpretation Value World) (module : Module) :
    ∀ depth, invoke meaning module false depth = invoke meaning (swapModule left right module) false depth := by
  intro depth
  induction depth with
  | zero => rfl
  | succ depth induction =>
    funext name
    simp only [invoke, functionAt_swapped]
    cases found : functionAt module name with
    | none => rfl
    | some function =>
      simp only [Option.map_some]
      rw [induction, withCalls_swapped]
      exact function_swapped left right function _

theorem complete_swapped {Value World : Type} (left right : Nat) (meaning : Interpretation Value World)
    (module : Module) (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    complete meaning module false name inputs before outputs after ↔
      complete meaning (swapModule left right module) false name inputs before outputs after := by
  unfold complete
  apply exists_congr
  intro depth
  rw [invoke_swapped]

end Merlin.GraphLoopsShape.After
