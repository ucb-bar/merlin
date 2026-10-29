import Merlin.GraphStateIntegerOperations
namespace Merlin.GraphScheduleObserverCountermodel
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
def before : Program := (.pure (.add 8) [0,1] (.pure (.mul 8) [1,2] (.opaque 0 [1] [(.bits 8)] (.pure (.sub 8) [3,4] (.opaque 1 [2] [(.bits 8)] (.pure (.add 8) [2,0] (.pure (.add 8) [0,2] (.ret [0]))))))))
def after : Program := (.pure (.add 8) [0,1] (.pure (.mul 8) [1,2] (.opaque 1 [0] [(.bits 8)] (.pure (.sub 8) [3,4] (.opaque 0 [3] [(.bits 8)] (.pure (.add 8) [0,2] (.pure (.add 8) [0,2] (.ret [0]))))))))
def chosen : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity arguments world =>
    if identity = 0 then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_failure : evalProgram chosen before [3,5] 7 = .error (.external 17,8) := rfl
theorem reordered_failure : evalProgram chosen after [3,5] 7 = .error (.external 17,9) := rfl
theorem terminal_world_differs : evalProgram chosen before [3,5] 7 ≠ evalProgram chosen after [3,5] 7 := by
  rw [source_failure,reordered_failure]
  decide
end Merlin.GraphScheduleObserverCountermodel
