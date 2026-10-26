/- Operational SHAPE/SSA initialization consequences of actual static typing.
Opaque calls may still fail; generic values may contain poison. This theorem
does not assert successful source execution, numerical fidelity or no poison. -/
import Merlin.GraphStateScalarSound
import Merlin.GraphStateScalarTyping

namespace Merlin.GraphState.Scalar
set_option autoImplicit false

def Shaped {Value World Error : Type} (count : Nat) : Outcome Value World Error → Prop
  | .ok (values,_) => values.length = count
  | .error (.external _,_) => True
  | .error (.arity,_) => False

theorem lookupSome_bound {α : Type} {values : List α} {index : Nat} {value : α}
    (found : values[index]? = some value) : index < values.length := by
  induction values generalizing index with
  | nil => cases found
  | cons head tail ih =>
    cases index with
    | zero => exact Nat.zero_lt_succ _
    | succ index => exact Nat.succ_lt_succ (ih found)

theorem lookup_inside {α : Type} {values : List α} {index : Nat} (inside : index < values.length) :
    ∃ value, values[index]? = some value := by
  induction values generalizing index with
  | nil => cases inside
  | cons head tail ih =>
    cases index with
    | zero => exact ⟨head,rfl⟩
    | succ index => exact ih (Nat.lt_of_succ_lt_succ inside)

theorem HasReads.initialized {types : List Ty} {indices : List Nat} {selected : List Ty}
    (typed : HasReads types indices selected) {Value : Type} (values : List Value)
    (arity : values.length = types.length) : ∀ index ∈ indices, ∃ value, values[index]? = some value := by
  induction typed with
  | nil => simp
  | @cons index type indices selected found rest ih =>
    intro actual member
    rcases List.mem_cons.mp member with same | later
    · subst actual
      apply lookup_inside
      rw [arity]
      exact lookupSome_bound found
    · exact ih actual later

theorem loopEval_shaped {Value World Error : Type} (count : Nat)
    (body : Int → List Value → World → Outcome Value World Error)
    (admitted : ∀ index values world, values.length = count → Shaped count (body index values world))
    (iterations : List Int) (values : List Value) (world : World) (arity : values.length = count) :
    Shaped count (loopEval count body iterations values world) := by
  induction iterations generalizing values world with
  | nil => simpa only [loopEval,ite_eq_left arity,Shaped] using arity
  | cons index later ih =>
    simp only [loopEval,ite_eq_left arity]
    have current := admitted index values world arity
    cases step : body index values world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      cases error with
      | external error => trivial
      | arity => simp only [step,Shaped] at current
    | ok next =>
      have size : next.1.length = count := by simpa only [step,Shaped] using current
      exact ih next.1 next.2 size

theorem typed_execution_shaped {registry : List Signature} {types results : List Ty} {program : Program}
    (typed : Typed registry types results program) {Value World Error : Type}
    (model : Model Value World Error registry) (values : List Value) (world : World)
    (arity : values.length = types.length) : Shaped results.length (evalProgram model program values world) := by
  induction typed generalizing values world with
  | ret operands =>
    simpa only [evalProgram,Shaped,reads,List.length_map] using operands.length
  | pure eligible operands later ih =>
    apply ih
    simpa only [List.length_cons] using congrArg Nat.succ arity
  | @effect types results outputs identity arguments selected next known equal operands later ih =>
    simp only [evalProgram]
    cases step : opaqueStep model identity (reads model.default values arguments) world with
    | error failure =>
      cases source : model.effect identity (reads model.default values arguments) world with
      | error failed =>
        rcases failed with ⟨error,final⟩
        simp only [opaqueStep,source] at step
        cases step
        trivial
      | ok result => simp only [opaqueStep,source] at step; cases step
    | ok next =>
      apply ih
      have size := opaqueStep_arity model step
      simp only [signature,known,Option.getD_some] at size
      simp only [List.length_append]
      rw [arity,size,equal]
  | @loop types results carries lower upper stride initial body next bounds operands inside later ihBody ihNext =>
    simp only [evalProgram]
    have bodyShape : ∀ index carry state, carry.length = carries.length →
        Shaped carries.length (evalProgram model body (model.index index::carry++values) state) := by
      intro index carry state size
      apply ihBody
      simp only [List.length_cons,List.length_append,size,arity]
    have initialSize : (reads model.default values initial).length = carries.length := by
      simpa only [reads,List.length_map] using operands.length
    have shape := loopEval_shaped carries.length _ bodyShape
      (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default) (values[stride]?.getD model.default))
      (reads model.default values initial) world initialSize
    cases outcome : loopEval carries.length
        (fun index carry state => evalProgram model body (model.index index::carry++values) state)
        (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default) (values[stride]?.getD model.default))
        (reads model.default values initial) world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      cases error with
      | external error => trivial
      | arity => simp only [outcome,Shaped] at shape
    | ok result =>
      apply ihNext
      have size : result.1.length = carries.length := by simpa only [outcome,Shaped] using shape
      simp only [List.length_append,size,arity]

def Present {Value : Type} (environment : List Value) (arguments : List Nat) : Prop :=
  ∀ index ∈ arguments, ∃ value, environment[index]? = some value

def loopInitialized {Value World Error : Type} (count : Nat)
    (safe : Int → List Value → World → Prop)
    (body : Int → List Value → World → Outcome Value World Error) : List Int → List Value → World → Prop
  | [],values,_ => values.length = count
  | index::later,values,world => values.length = count ∧ safe index values world ∧
    match body index values world with
    | .error _ => True
    | .ok (next,final) => loopInitialized count safe body later next final

def Initialized {Value World Error : Type} {registry : List Signature}
    (model : Model Value World Error registry) : Program → List Value → World → Prop
  | .ret arguments,environment,_ => Present environment arguments
  | .pure op arguments next,environment,world => Present environment arguments ∧
    Initialized model next (model.primitive op (reads model.default environment arguments)::environment) world
  | .opaque identity arguments _ next,environment,world => Present environment arguments ∧
    match opaqueStep model identity (reads model.default environment arguments) world with
    | .error _ => True
    | .ok (values,final) => Initialized model next (values++environment) final
  | .loop lower upper stride initial carries body next,environment,world =>
    Present environment [lower,upper,stride] ∧ Present environment initial ∧
    loopInitialized carries.length
      (fun index values current => Initialized model body (model.index index::values++environment) current)
      (fun index values current => evalProgram model body (model.index index::values++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world ∧
    match loopEval carries.length
      (fun index values current => evalProgram model body (model.index index::values++environment) current)
      (model.iterations (environment[lower]?.getD model.default) (environment[upper]?.getD model.default)
        (environment[stride]?.getD model.default)) (reads model.default environment initial) world with
    | .error _ => True
    | .ok (values,final) => Initialized model next (values++environment) final

theorem loop_initialized {Value World Error : Type} (count : Nat)
    (safe : Int → List Value → World → Prop) (body : Int → List Value → World → Outcome Value World Error)
    (bodySafe : ∀ index values world, values.length = count → safe index values world)
    (bodyShaped : ∀ index values world, values.length = count → Shaped count (body index values world))
    (iterations : List Int) (values : List Value) (world : World) (arity : values.length = count) :
    loopInitialized count safe body iterations values world := by
  induction iterations generalizing values world with
  | nil => exact arity
  | cons index later ih =>
    refine ⟨arity,bodySafe index values world arity,?_⟩
    have shaped := bodyShaped index values world arity
    cases actual : body index values world with
    | error failure => trivial
    | ok next =>
      apply ih next.1 next.2
      simpa only [actual,Shaped] using shaped

/-- Every actually reached operand/capture/carry read resolves to an initialized
SSA value; error paths stop rather than assuming later computations succeed. -/
theorem typed_operands_initialized {registry : List Signature} {types results : List Ty} {program : Program}
    (typed : Typed registry types results program) {Value World Error : Type}
    (model : Model Value World Error registry) (values : List Value) (world : World)
    (arity : values.length = types.length) : Initialized model program values world := by
  induction typed generalizing values world with
  | ret operands => exact operands.initialized values arity
  | pure eligible operands later ih =>
    refine ⟨operands.initialized values arity,?_⟩
    apply ih
    simpa only [List.length_cons] using congrArg Nat.succ arity
  | @effect types results outputs identity arguments selected next known equal operands later ih =>
    refine ⟨operands.initialized values arity,?_⟩
    cases step : opaqueStep model identity (reads model.default values arguments) world with
    | error failure => trivial
    | ok next =>
      apply ih
      have size := opaqueStep_arity model step
      simp only [signature,known,Option.getD_some] at size
      simp only [List.length_append]
      rw [arity,size,equal]
  | @loop types results carries lower upper stride initial body next bounds operands inside later ihBody ihNext =>
    refine ⟨bounds.initialized values arity,operands.initialized values arity,?_,?_⟩
    · apply loop_initialized
      · intro index carry current size
        apply ihBody
        simp only [List.length_cons,List.length_append,size,arity]
      · intro index carry current size
        apply typed_execution_shaped inside model
        simp only [List.length_cons,List.length_append,size,arity]
      · simpa only [reads,List.length_map] using operands.length
    · cases outcome : loopEval carries.length
          (fun index carry current => evalProgram model body (model.index index::carry++values) current)
          (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
            (values[stride]?.getD model.default)) (reads model.default values initial) world with
      | error failure => trivial
      | ok next =>
        apply ihNext
        have size := loopEval_arity carries.length _ _ _ _ outcome
        simp only [List.length_append,size,arity]

#print axioms typed_execution_shaped
#print axioms HasReads.initialized
#print axioms typed_operands_initialized
end Merlin.GraphState.Scalar
