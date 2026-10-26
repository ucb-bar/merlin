/- Complete fixed-profile index literal domain, independently checked.
This checks all control-classified constants, including captured/ghost values;
W's separate bounds discharge generated add/min/IV increment obligations. -/
import Merlin.GraphStatePipelineIndex

namespace Merlin.GraphState.Pipeline
set_option autoImplicit false

instance (bits : Nat) (value : Int) : Decidable (signedRepresentative bits value) := by
  unfold signedRepresentative
  infer_instance

def checkLiterals (bits : Nat) (controls : Controls) : Bool :=
  controls.all (fun entry => match entry with
    | some (.constant value) => decide (signedRepresentative bits value)
    | _ => true)

theorem checkLiterals_sound {bits : Nat} {controls : Controls}
    (accepted : checkLiterals bits controls = true) {identity : Nat} {value : Int}
    (found : controlAt controls identity = some (.constant value)) : signedRepresentative bits value := by
  have member : some (IndexOp.constant value) ∈ controls := by
    unfold controlAt at found
    cases item : controls[identity]? with
    | none => simp only [item,Option.getD_none] at found; cases found
    | some entry =>
      simp only [item,Option.getD_some] at found
      subst entry
      exact List.mem_of_getElem? item
  have checked := List.all_eq_true.mp accepted (some (.constant value)) member
  exact of_decide_eq_true checked

end Merlin.GraphState.Pipeline
