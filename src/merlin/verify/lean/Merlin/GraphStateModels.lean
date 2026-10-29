/- Reusable total framed actions and explicit nonvacuity witnesses. Arithmetic
is an unchanged arbitrary total function; this creates no numerical equality or
IR operational bridge. Actual-instance rosters must be separately matched. -/
import Merlin.GraphStateConfiguration

namespace Merlin.GraphState.Model
open StateEffects
set_option autoImplicit false

def writeAction {Value : Type} (inputs : List Nat) (output : Nat)
    (compute : List Value → Value) : Action ⟨inputs,[output]⟩ Value where
  step := fun state key => if key = output then compute (inputs.map state) else state key
  frame := by
    intro state key outside
    have different : key ≠ output := by simpa using outside
    simp [different]
  locality := by
    intro first second agree key inside
    have equal : inputs.map first = inputs.map second := by
      apply List.map_congr_left
      exact agree
    have same : key = output := by simpa using inside
    simp [same,equal]

def identityAction {Value : Type} (inputs : List Nat) : Action ⟨inputs,[]⟩ Value where
  step := id
  frame := by intros; rfl
  locality := by intro _ _ _ _ impossible; cases impossible

structure Instruction (Value : Type) where
  effects : Footprint
  action : Action effects Value

def run {Value : Type} (roster : List (Instruction Value)) (index : Nat) : State Value → State Value :=
  match roster[index]? with
  | none => id
  | some instruction => instruction.action.step

theorem roster_models {Value : Type} (roster : List (Instruction Value)) :
    Models (roster.map (·.effects)) (run roster) := by
  intro index effects found
  cases actual : roster[index]? with
  | none => simp [List.getElem?_map,actual] at found
  | some instruction =>
    have same : instruction.effects = effects := by
      simpa only [List.getElem?_map,actual,Option.map_some,Option.some.injEq] using found
    subst effects
    exact ⟨instruction.action,by simp only [run,actual]⟩

#print axioms roster_models
#print axioms writeAction
end Merlin.GraphState.Model
