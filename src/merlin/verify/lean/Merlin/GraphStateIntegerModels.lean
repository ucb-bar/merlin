/- Inhabited initialized mathematical models. Success is DERIVED from typed
syntax and local total-on-typed call behavior, not supplied as a source premise.
An optional faulty provider remains explicitly partial with a terminal world. -/
import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar Pipeline

def TotalOnTyped {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (rep : Ty → Int → Prop) : Prop :=
  ∀ identity arguments world, Values rep (signature registry identity).inputs arguments →
    ∃ values final, model.effect identity arguments world = .ok (values,final)

theorem loop_success {World Error : Type} {rep : Ty → Int → Prop} (types : List Ty)
    (body : Int → List Int → World → Outcome Int World Error)
    (total : ∀ index values world, Values rep types values →
      ∃ outputs final, body index values world = .ok (outputs,final) ∧ Values rep types outputs)
    (iterations : List Int) (values : List Int) (world : World) (typed : Values rep types values) :
    ∃ outputs final, loopEval types.length body iterations values world = .ok (outputs,final) ∧
      Values rep types outputs := by
  induction iterations generalizing values world with
  | nil => exact ⟨values,world,by simp only [loopEval,typed.length,↓reduceIte],typed⟩
  | cons index later ih =>
    obtain ⟨next,current,step,nextTyped⟩ := total index values world typed
    obtain ⟨outputs,final,rest,outputTyped⟩ := ih next current nextTyped
    exact ⟨outputs,final,by simp only [loopEval,typed.length,↓reduceIte,step,Except.bind,rest],outputTyped⟩

theorem typed_success {World Error : Type} {registry : List Signature} {rep : Ty → Int → Prop}
    (model : Model Int World Error registry) (laws : LocalTyping model rep) (total : TotalOnTyped model rep)
    {types results : List Ty} {program : Program} (typed : Typed registry types results program)
    (values : List Int) (world : World) (entry : Values rep types values) :
    ∃ outputs final, evalProgram model program values world = .ok (outputs,final) ∧ Values rep results outputs := by
  induction typed generalizing values world with
  | ret operands => exact ⟨_,world,rfl,entry.reads operands model.default⟩
  | pure eligible operands later ih =>
    exact ih _ world (.cons (laws.primitive _ _ eligible (entry.reads operands model.default)) entry)
  | @effect types results outputs identity arguments selected next known equal operands later ih =>
    have inputs : Values rep (signature registry identity).inputs (reads model.default values arguments) := by
      simpa only [signature,known,Option.getD_some] using entry.reads operands model.default
    obtain ⟨returned,current,step⟩ := total identity _ world inputs
    have typedOutputs := laws.effect _ _ _ _ _ inputs step
    have same : (signature registry identity).outputs = outputs := by simp [signature,known,equal]
    rw [same] at typedOutputs
    obtain ⟨result,final,success,resultTyped⟩ := ih _ current (typedOutputs.append entry)
    exact ⟨result,final,by simp only [evalProgram,opaqueStep,step,Except.bind,success],resultTyped⟩
  | @loop types results carries lower upper stride initial body next bounds operands inside later ihBody ihNext =>
    obtain ⟨returned,current,step,typedOutputs⟩ := loop_success carries
      (fun index carry current => evalProgram model body (model.index index::carry++values) current)
      (fun index carry current carryTyped => ihBody _ current
        (.cons (laws.index index) (carryTyped.append entry)))
      (model.iterations (values[lower]?.getD model.default) (values[upper]?.getD model.default)
        (values[stride]?.getD model.default)) (reads model.default values initial) world
      (entry.reads operands model.default)
    obtain ⟨result,final,success,resultTyped⟩ := ihNext _ current (typedOutputs.append entry)
    exact ⟨result,final,by simp only [evalProgram,step,Except.bind,success],resultTyped⟩

def inhabit : Ty → Int
  | .bits width => word width 17
  | .index | .opaque _ => 17

theorem inhabit_typed (type : Ty) : Canonical type (inhabit type) := by
  cases type <;> simp only [inhabit,Canonical,word_word] <;> trivial

theorem inhabitants (types : List Ty) : Values Canonical types (types.map inhabit) := by
  induction types with
  | nil => exact .nil
  | cons type later ih => exact .cons (inhabit_typed type) ih

def provider (registry : List Signature) (failure : Option Nat) : External Nat Int registry where
  run := fun identity _ world =>
    if failure = some identity then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map inhabit,world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]

theorem provider_typed (registry : List Signature) (failure : Option Nat)
    (identity : Nat) (arguments : List Int) (world : Nat) (values : List Int) (final : Nat)
    (_inputs : Values Canonical (signature registry identity).inputs arguments)
    (success : (provider registry failure).run identity arguments world = .ok (values,final)) :
    Values Canonical (signature registry identity).outputs values := by
  simp only [provider] at success
  split at success
  · cases success
  · cases success; exact inhabitants _

theorem provider_total {registry : List Signature} {controls : Controls} (valid : ControlValid registry controls) :
    TotalOnTyped (indexModel valid primitive (provider registry none)) Canonical := by
  intro identity arguments world inputs
  cases found : controlAt controls identity with
  | none => exact ⟨(signature registry identity).outputs.map inhabit,world+1,
      by simp [indexModel,indexEffect,found,provider]⟩
  | some op => exact ⟨[op.eval arguments],world,by simp only [indexModel,indexEffect,found]⟩

/-- Constant-result/effect-count loop witnesses do not expand a dynamic trace. -/
theorem loop_constant (value : Int) (increment : Nat)
    (body : Int → List Int → Nat → Outcome Int Nat Int)
    (bodyExact : ∀ index carry world, carry.length = 1 → body index carry world = .ok ([value],world+increment))
    (first : Int) (later : List Int) (carry : List Int) (world : Nat) (arity : carry.length = 1) :
    loopEval 1 body (first::later) carry world = .ok ([value],world+increment*(later.length+1)) := by
  induction later generalizing first carry world with
  | nil => simp [loopEval,arity,bodyExact first carry world arity,Except.bind]
  | cons index rest ih =>
    rw [loopEval,ite_eq_left arity,bodyExact first carry world arity]
    change loopEval 1 body (index::rest) [value] (world+increment) = _
    rw [ih index [value] (world+increment) rfl]
    congr 2
    simp only [List.length_cons,Nat.mul_add,Nat.mul_one]
    omega

#print axioms typed_success
#print axioms provider_typed
#print axioms loop_constant
end Merlin.GraphState.Integer
