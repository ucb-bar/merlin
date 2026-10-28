/- Choose common scalar evidence by checking every source refinement. Candidate generation
is independent of soundness: future joins can supply additional candidates without asserting
their correctness. Failure to establish a common fact falls back to the model's empty fact. -/
import MerlinVerify.Check.DefinedContextGeneric

namespace MerlinVerify.Check.Definedness.Refined.Common

open MerlinVerify

variable {Info : Type} {M : Model Info} {ib : Nat}

def scalar (elt : Ty) (info : Info) : Fact Info := ⟨⟨.scalar elt, true⟩, info⟩

def select (rules : Contextual.Structured.Invariants M ib) (elt : Ty) (inputs : List Info) : List Info → Info
  | [] => M.empty
  | candidate :: rest =>
    if inputs.all (fun info => rules.implies [scalar elt info] [scalar elt candidate]) then candidate
    else select rules elt inputs rest

theorem select_sound (rules : Contextual.Structured.Invariants M ib) (elt : Ty) (inputs : List Info)
    (candidates : List Info) {info : Info} (member : info ∈ inputs) {value : Val}
    (holds : (scalar elt info).Holds M ib (.scalar value)) :
    M.holds (select rules elt inputs candidates) (.scalar value) := by
  induction candidates with
  | nil => exact M.empty_sound _
  | cons candidate rest ih =>
    unfold select
    split
    · next checked =>
      have accepted := List.all_eq_true.1 checked info member
      have outputs := rules.implies_sound _ _ accepted _ (.cons holds .nil)
      cases outputs with
      | cons head _ => exact head.2
    · exact ih

def infer (rules : Contextual.Structured.Invariants M ib) (elt : Ty) (inputs : List Info) : Info :=
  select rules elt inputs (inputs ++ inputs.map rules.widen)

theorem infer_sound (rules : Contextual.Structured.Invariants M ib) (elt : Ty) (inputs : List Info)
    {info : Info} (member : info ∈ inputs) {value : Val}
    (holds : (scalar elt info).Holds M ib (.scalar value)) :
    M.holds (infer rules elt inputs) (.scalar value) := select_sound rules elt inputs _ member holds

#print axioms select_sound
#print axioms infer_sound

end MerlinVerify.Check.Definedness.Refined.Common
