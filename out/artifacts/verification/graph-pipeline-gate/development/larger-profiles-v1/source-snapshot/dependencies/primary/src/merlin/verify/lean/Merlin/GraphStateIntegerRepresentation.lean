/- Runtime value typing for the EXISTING Scalar operational evaluator.
Local result contracts plus typed canonical entries derive every successful
result/carry representation. They do not assume source success or no poison
for a physical execution; the admitted carrier contains initialized Ints. -/
import Merlin.GraphStateScalar

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar

def word (width : Nat) (value : Int) : Int := (BitVec.ofInt width value).toNat

def Canonical : Ty → Int → Prop
  | .bits width,value => word width value = value
  | .index,_ | .opaque _,_ => True

inductive Values (rep : Ty → Int → Prop) : List Ty → List Int → Prop where
  | nil : Values rep [] []
  | cons {type : Ty} {types : List Ty} {value : Int} {values : List Int}
      (head : rep type value) (tail : Values rep types values) :
      Values rep (type::types) (value::values)

theorem Values.length {rep : Ty → Int → Prop} {types : List Ty} {values : List Int}
    (typed : Values rep types values) : values.length = types.length := by
  induction typed with
  | nil => rfl
  | cons _ _ ih => simpa using congrArg Nat.succ ih

theorem Values.append {rep : Ty → Int → Prop} {first second : List Ty} {xs ys : List Int}
    (left : Values rep first xs) (right : Values rep second ys) :
    Values rep (first++second) (xs++ys) := by
  induction left with
  | nil => exact right
  | cons head tail ih => exact .cons head ih

theorem Values.lookup {rep : Ty → Int → Prop} {types : List Ty} {values : List Int}
    (typed : Values rep types values) {index : Nat} {type : Ty}
    (found : types[index]? = some type) (default : Int) :
    rep type (values[index]?.getD default) := by
  induction typed generalizing index with
  | nil => cases found
  | cons head tail ih =>
    cases index with
    | zero => cases found; exact head
    | succ index => exact ih found

theorem Values.reads {rep : Ty → Int → Prop} {types selected : List Ty} {values : List Int}
    {indices : List Nat} (typed : Values rep types values)
    (operands : HasReads types indices selected) (default : Int) :
    Values rep selected (reads default values indices) := by
  induction operands with
  | nil => exact .nil
  | cons found rest ih => exact .cons (typed.lookup found default) ih

/-- These laws are LOCAL to individual admitted operations. An opaque failure
has no result-representation obligation and retains its entire error/world. -/
structure LocalTyping {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (rep : Ty → Int → Prop) : Prop where
  primitive : ∀ op arguments, op.valid = true → Values rep op.inputs arguments →
    rep op.output (model.primitive op arguments)
  index : ∀ index, rep .index (model.index index)
  effect : ∀ identity arguments world values final,
    Values rep (signature registry identity).inputs arguments →
    model.effect identity arguments world = .ok (values,final) →
    Values rep (signature registry identity).outputs values

def ResultTyped {World Error : Type} (rep : Ty → Int → Prop) (types : List Ty) :
    Outcome Int World Error → Prop
  | .ok (values,_) => Values rep types values
  | .error (.external _,_) => True
  | .error (.arity,_) => False

theorem loop_result_typed {World Error : Type} {rep : Ty → Int → Prop} (types : List Ty)
    (body : Int → List Int → World → Outcome Int World Error)
    (bodyTyped : ∀ index values world, Values rep types values → ResultTyped rep types (body index values world))
    (iterations : List Int) (values : List Int) (world : World) (typed : Values rep types values) :
    ResultTyped rep types (loopEval types.length body iterations values world) := by
  induction iterations generalizing values world with
  | nil => simpa only [loopEval,typed.length,↓reduceIte,ResultTyped] using typed
  | cons index rest ih =>
    simp only [loopEval,typed.length,↓reduceIte]
    have current := bodyTyped index values world typed
    cases step : body index values world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      cases error with
      | external error => trivial
      | arity => simp only [step,ResultTyped] at current
    | ok next =>
      exact ih next.1 next.2 (by simpa only [step,ResultTyped] using current)

/-- Runtime result representations, including recursively carried values, are
derived from static syntax and LOCAL contracts. This is conditional on canonical
entries, not a global intermediate-typing or successful-execution hypothesis. -/
theorem typed_result {World Error : Type} {registry : List Signature}
    {rep : Ty → Int → Prop} (model : Model Int World Error registry) (laws : LocalTyping model rep)
    {types results : List Ty} {program : Program} (typed : Typed registry types results program)
    (values : List Int) (world : World) (entry : Values rep types values) :
    ResultTyped rep results (evalProgram model program values world) := by
  induction typed generalizing values world with
  | ret operands => exact entry.reads operands model.default
  | pure eligible operands later ih =>
    exact ih _ world (.cons (laws.primitive _ _ eligible (entry.reads operands model.default)) entry)
  | @effect types results outputs identity arguments selected next known equal operands later ih =>
    simp only [evalProgram,opaqueStep]
    cases step : model.effect identity (reads model.default values arguments) world with
    | error failure => trivial
    | ok result =>
      apply ih
      apply Values.append _ entry
      have inputs : (signature registry identity).inputs = selected.inputs := by simp [signature,known]
      have outputs : (signature registry identity).outputs = outputs := by simp [signature,known, equal]
      rw [← outputs]
      exact laws.effect _ _ _ _ _ (by rw [inputs]; exact entry.reads operands model.default) step
  | @loop types results carries lower upper stride initial body next bounds operands inside later ihBody ihNext =>
    simp only [evalProgram]
    have shape := loop_result_typed carries
      (fun index carry current => evalProgram model body (model.index index::carry++values) current)
      (fun index carry current carryTyped => ihBody _ current
        (.cons (laws.index index) (carryTyped.append entry)))
      (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
        (values[stride]?.getD model.default)) (reads model.default values initial) world
      (entry.reads operands model.default)
    cases outcome : loopEval carries.length
        (fun index carry current => evalProgram model body (model.index index::carry++values) current)
        (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
          (values[stride]?.getD model.default)) (reads model.default values initial) world with
    | error failure =>
      rcases failure with ⟨error,final⟩
      cases error with
      | external error => trivial
      | arity => simp only [outcome,ResultTyped] at shape
    | ok result =>
      apply ihNext
      exact (show Values rep carries result.1 from by simpa only [outcome,ResultTyped] using shape).append entry

#print axioms typed_result
#print axioms loop_result_typed
end Merlin.GraphState.Integer
