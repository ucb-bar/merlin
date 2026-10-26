/- Whole-module scheduling acceptance, reusing the sealed Scalar normalizer.
Eligible Prim meanings are TOTAL/state-free by the Model type; empty host
effects are not an extra totality theorem. Opaque steps/regions stay anchored
in the independently checked complete Code skeleton and host IR permutation.
No topological plan or supplied permutation is an execution proof. -/
import Merlin.GraphStateScalar
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStateIntegerProgram
import Merlin.GraphStateIntegerTrace

namespace Merlin.GraphState.Schedule
set_option autoImplicit false
open Scalar Pipeline

def checkPair (registry : List Signature) : List Entry → List Entry → Bool
  | [],[] => true
  | first::before,second::after =>
    decide (first.inputs = second.inputs) && decide (first.outputs = second.outputs) &&
    Scalar.check registry first.inputs first.outputs first.body second.body && checkPair registry before after
  | _,_ => false

theorem checkPair_sound {registry : List Signature} {before after : List Entry}
    (accepted : checkPair registry before after = true) {Value World Error : Type}
    (model : Model Value World Error registry) : PreservesEntries model before after := by
  induction before generalizing after with
  | nil => cases after with
    | nil => exact Pipeline.entries_nil model
    | cons _ _ => cases accepted
  | cons first before ih =>
    cases after with
    | nil => cases accepted
    | cons second after =>
      have parts : ((first.inputs = second.inputs ∧ first.outputs = second.outputs) ∧
          Scalar.check registry first.inputs first.outputs first.body second.body = true) ∧
          checkPair registry before after = true := by
        simpa only [checkPair,Bool.and_eq_true,decide_eq_true_eq] using accepted
      apply Pipeline.entries_cons model parts.1.1.1 parts.1.1.2
      · intro arguments world arity
        exact Scalar.check_sound parts.1.2 Value World Error model arguments world arity
      · exact ih parts.2

/-- Every actual requested order supplies a complete module pair. Later source
buffers are the current program, not a stale original planner witness. -/
def checkSteps (registry : List Signature) (before : List Entry) :
    List (List Entry) → List Entry → Bool
  | [],after => checkPair registry before after
  | middle::later,after => checkPair registry before middle && checkSteps registry middle later after

theorem checkSteps_sound {registry : List Signature} {before after : List Entry}
    (middles : List (List Entry)) (accepted : checkSteps registry before middles after = true)
    {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model before after := by
  induction middles generalizing before with
  | nil => exact checkPair_sound accepted model
  | cons middle later ih =>
    have parts : checkPair registry before middle = true ∧ checkSteps registry middle later after = true := by
      simpa only [checkSteps,Bool.and_eq_true] using accepted
    intro identity arguments world
    exact (checkPair_sound parts.1 model identity arguments world).trans
      (ih parts.2 identity arguments world)

/-- Value typing remains a separately derived consequence of local contracts;
the general scheduling theorem needs no canonical-value numerical identity. -/
theorem checked_reached {World Error : Type} {registry : List Signature}
    {inputs outputs : List Ty} {before after : Program}
    (accepted : Scalar.check registry inputs outputs before after = true)
    (model : Model Int World Error registry) (laws : Integer.LocalTyping model Integer.Canonical)
    (values : List Int) (world : World) (entry : Integer.Values Integer.Canonical inputs values) :
    Integer.Reached model Integer.Canonical before values world ∧
      Integer.Reached model Integer.Canonical after values world :=
  ⟨Integer.typed_reached model laws (Scalar.check_typed accepted).1 values world entry,
   Integer.typed_reached model laws (Scalar.check_typed accepted).2 values world entry⟩

/-- Exact whole-program join is supplied by checked syntax in each query. This
composes canonical integer/loop prefixes with generic total-primitive scheduling. -/
theorem compose_prefix {World Error : Type} {registry : List Signature}
    (model : Model Int World Error registry) {before middle after : List Entry}
    (earlier : Integer.PreservesEntriesCanonical model before middle)
    (accepted : checkPair registry middle after = true) :
    Integer.PreservesEntriesCanonical model before after :=
  Integer.compose_suffix model earlier (checkPair_sound accepted model)

#print axioms checkPair_sound
#print axioms checkSteps_sound
#print axioms checked_reached
#print axioms compose_prefix
end Merlin.GraphState.Schedule
