/- Complete entry-roster correspondence on a canonical entry domain, using the
UNCHANGED pipeline dispatcher. Wrong arity and missing entries remain visible.
The suffix may be the sealed four-pass theorem in the SAME actual registry. -/
import Merlin.GraphStateIntegerCode
import Merlin.GraphStatePipelineProgram

namespace Merlin.GraphState.Integer
set_option autoImplicit false
open Scalar Pipeline

def EntryDomain (entries : List Entry) (identity : Nat) (arguments : List Int) : Prop :=
  match entries[identity]? with
  | none => True
  | some entry => arguments.length ≠ entry.inputs.length ∨ Values Canonical entry.inputs arguments

def PreservesEntriesCanonical {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) (before after : List Entry) : Prop :=
  ∀ identity arguments world, EntryDomain before identity arguments →
    evalEntry model before identity arguments world = evalEntry model after identity arguments world

theorem entries_nil {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) : PreservesEntriesCanonical model [] [] := by
  intro identity arguments world entry
  rfl

theorem entries_cons {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) {first second : Entry} {before after : List Entry}
    (inputs : first.inputs = second.inputs) (_outputs : first.outputs = second.outputs)
    (body : ∀ arguments world, Values Canonical first.inputs arguments →
      evalProgram model first.body arguments world = evalProgram model second.body arguments world)
    (rest : PreservesEntriesCanonical model before after) :
    PreservesEntriesCanonical model (first::before) (second::after) := by
  intro identity arguments world entry
  cases identity with
  | zero =>
    simp only [evalEntry,List.getElem?_cons_zero]
    rw [← inputs]
    by_cases arity : arguments.length = first.inputs.length
    · have canonical : Values Canonical first.inputs arguments := by
        change arguments.length ≠ first.inputs.length ∨ Values Canonical first.inputs arguments at entry
        exact entry.resolve_left (fun wrong => wrong arity)
      simp only [arity,↓reduceIte,body arguments world canonical]
    · simp only [arity,↓reduceIte]
  | succ identity =>
    exact rest identity arguments world (by simpa only [EntryDomain,List.getElem?_cons_succ] using entry)

theorem compose_suffix {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) {before middle after : List Entry}
    (earlier : PreservesEntriesCanonical model before middle)
    (suffix : PreservesEntries model middle after) : PreservesEntriesCanonical model before after := by
  intro identity arguments world entry
  exact (earlier identity arguments world entry).trans (suffix identity arguments world)

/-- Opaque successful results require their LOCAL representation contract.
Index operations return the concrete index carrier and impose no word premise. -/
theorem index_model_typing {World Error : Type} {registry : List Signature} {controls : Controls}
    (valid : ControlValid registry controls) (external : External World Error registry)
    (typed : ∀ identity arguments world values final,
      Values Canonical (signature registry identity).inputs arguments →
      external.run identity arguments world = .ok (values,final) →
      Values Canonical (signature registry identity).outputs values) :
    LocalTyping (indexModel valid primitive external) Canonical where
  primitive := fun op arguments _ _ => primitive_canonical op arguments
  index := fun _ => True.intro
  effect := by
    intro identity arguments world values final inputs success
    cases found : controlAt controls identity with
    | none => exact typed _ _ _ _ _ inputs (by simpa only [indexModel,indexEffect,found] using success)
    | some op =>
      have same : ([op.eval arguments],world) = (values,final) := by
        simpa only [indexModel,indexEffect,found,Except.ok.injEq] using success
      cases same
      rw [(valid.2 identity op found).2.2]
      exact .cons True.intro .nil

#print axioms compose_suffix
#print axioms entries_cons
#print axioms index_model_typing
end Merlin.GraphState.Integer
