/- Partial retained-body counterpart of the frozen total configuration hoist.
Only successful steps maintain setup fixedness; arbitrary failures keep their
complete error/state unchanged by short-circuiting. No source success premise. -/
import Merlin.GraphStateElision

namespace Merlin.GraphState.Configuration.PartialHoist
open StateEffects
set_option autoImplicit false

abbrev Outcome (Value Error : Type) := Except (Error × State Value) (State Value)

def Frame {Value Error : Type} (fields : List Nat)
    (body : State Value → Outcome Value Error) : Prop :=
  ∀ initial final, body initial = .ok final →
    ∀ key ∈ fields, final key = initial key

theorem assign_fixed_after {Value : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (initial final : State Value)
    (fixed : Elision.assign literal writes initial = initial)
    (frame : ∀ key ∈ writes.map (·.1), final key = initial key) :
    Elision.assign literal writes final = final := by
  funext key
  cases found : writes.lookup key with
  | none => simp only [Elision.assign,found]
  | some value =>
    have member : key ∈ writes.map (·.1) :=
      List.mem_map.mpr ⟨(key,value),Elision.lookup_member found,rfl⟩
    have installed := congrFun fixed key
    simp only [Elision.assign,found] at installed
    simp only [Elision.assign,found]
    exact installed.trans (frame key member).symm

def repeatBody {Value Error : Type}
    (body : Nat → State Value → Outcome Value Error) : Nat → Nat → State Value → Outcome Value Error
  | 0,_,state => .ok state
  | count+1,index,state => (body index state).bind (repeatBody body count (index+1))

def repeatSetup {Value Error : Type} (setup : State Value → State Value)
    (body : Nat → State Value → Outcome Value Error) : Nat → Nat → State Value → Outcome Value Error
  | 0,_,state => .ok state
  | count+1,index,state => (body index (setup state)).bind (repeatSetup setup body count (index+1))

theorem remove_fixed_setups {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (body : Nat → State Value → Outcome Value Error)
    (frame : ∀ index, Frame (writes.map (·.1)) (body index)) :
    ∀ count index initial, Elision.assign literal writes initial = initial →
      repeatSetup (Elision.assign literal writes) body count index initial =
        repeatBody body count index initial := by
  intro count
  induction count with
  | zero => intros; rfl
  | succ count ih =>
    intro index initial fixed
    simp only [repeatSetup,repeatBody,fixed]
    cases step : body index initial with
    | error failure => rfl
    | ok final =>
      simp only [Except.bind]
      exact ih (index+1) final (assign_fixed_after literal writes initial final fixed
        (frame index initial final step))

/-- Retain the first setup before a positive loop, remove only later fixed
setups. A failing retained call has the identical error AND terminal state. -/
theorem hoist_nonempty_partial {Value Error : Type} (literal : Int → Value)
    (writes : Elision.Assignments) (body : Nat → State Value → Outcome Value Error)
    (frame : ∀ index, Frame (writes.map (·.1)) (body index))
    (count : Nat) (positive : count > 0) (index : Nat) (initial : State Value) :
    repeatSetup (Elision.assign literal writes) body count index initial =
      repeatBody body count index (Elision.assign literal writes initial) := by
  cases count with
  | zero => cases positive
  | succ count =>
    simp only [repeatSetup,repeatBody]
    cases step : body index (Elision.assign literal writes initial) with
    | error failure => rfl
    | ok final =>
      simp only [Except.bind]
      exact remove_fixed_setups literal writes body frame count (index+1) final
        (assign_fixed_after literal writes _ final (Elision.assign_idempotent literal writes initial)
          (frame index _ final step))

end Merlin.GraphState.Configuration.PartialHoist
