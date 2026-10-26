/- Necessity of anchoring partial actions/regions even when a host reports no
memory effects. This is an explicit mathematical contract countermodel, not
a claim that native divsi raises these errors or updates this world. -/
import Merlin.GraphStateSchedule
import Merlin.GraphStateIntegerOperations

namespace Merlin.GraphState.Schedule.Countermodels
set_option autoImplicit false
open Scalar

def registry : List Signature := [
  {inputs := [.bits 8],outputs := [.bits 8]},
  {inputs := [.bits 8,.bits 8],outputs := [.bits 8]}]

def model : Model Int Nat Int registry where
  default := 0
  primitive := Integer.primitive
  index := id
  iterations := fun _ _ _ => [0]
  effect := fun identity _ world =>
    if identity = 0 then .error (17,world+1) else .error (18,world+2)
  arity := by
    intro identity arguments world values final success
    split at success <;> cases success

def before : Program := .opaque 0 [0] [.bits 8] (.opaque 1 [1,2] [.bits 8] (.ret [1]))
def after : Program := .opaque 1 [0,1] [.bits 8] (.opaque 0 [1] [.bits 8] (.ret [0]))

theorem both_statically_typed :
    checkTypes registry [.bits 8,.bits 8] [.bits 8] before = true ∧
    checkTypes registry [.bits 8,.bits 8] [.bits 8] after = true := by decide

theorem first_failure : evalProgram model before [3,5] 7 = .error (.external 17,8) := rfl
theorem reordered_failure : evalProgram model after [3,5] 7 = .error (.external 18,9) := rfl

theorem partial_actions_cannot_be_reordered :
    evalProgram model before [3,5] 7 ≠ evalProgram model after [3,5] 7 := by
  rw [first_failure,reordered_failure]
  intro same
  have impossible : (8 : Nat) = 9 := congrArg Prod.snd (Except.error.inj same)
  omega

def body : Program := .opaque 1 [1,1] [.bits 8] (.ret [0])
def regionBefore : Program := .opaque 0 [3] [.bits 8]
  (.loop 1 2 3 [4] [.bits 8] body (.ret [0]))
def regionAfter : Program := .loop 0 1 2 [3] [.bits 8] body
  (.opaque 0 [4] [.bits 8] (.ret [0]))

theorem regions_statically_typed :
    checkTypes registry [.index,.index,.index,.bits 8,.bits 8] [.bits 8] regionBefore = true ∧
    checkTypes registry [.index,.index,.index,.bits 8,.bits 8] [.bits 8] regionAfter = true := by decide

theorem anchored_region_failure :
    evalProgram model regionBefore [0,1,1,3,5] 7 = .error (.external 17,8) := rfl
theorem moved_region_failure :
    evalProgram model regionAfter [0,1,1,3,5] 7 = .error (.external 18,9) := rfl

theorem region_totality_cannot_be_inferred_from_empty_effects :
    evalProgram model regionBefore [0,1,1,3,5] 7 ≠ evalProgram model regionAfter [0,1,1,3,5] 7 := by
  rw [anchored_region_failure,moved_region_failure]
  intro same
  have impossible : (8 : Nat) = 9 := congrArg Prod.snd (Except.error.inj same)
  omega

#print axioms partial_actions_cannot_be_reordered
#print axioms region_totality_cannot_be_inferred_from_empty_effects
end Merlin.GraphState.Schedule.Countermodels
