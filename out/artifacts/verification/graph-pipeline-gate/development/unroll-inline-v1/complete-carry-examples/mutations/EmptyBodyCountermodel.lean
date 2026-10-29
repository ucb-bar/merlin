import Merlin.GraphStateIntegerOperations
import Merlin.GraphStatePipelineControl
namespace Merlin.GraphInlineEmptyCountermodel
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [.index,(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8),(.bits 8)],outputs := [(.bits 8)]}]
def source : Program := (.opaque 1 [] [.index] (.opaque 1 [] [.index] (.opaque 0 [] [.index] (.loop 2 1 0 [3,4] [(.bits 8),(.bits 8)] (.pure (.add 8) [1,7] (.opaque 2 [1,0] [(.bits 8)] (.ret [0,4]))) (.opaque 3 [0,1] [(.bits 8)] (.ret [0]))))))
def target : Program := (.opaque 1 [] [.index] (.opaque 1 [] [.index] (.opaque 0 [] [.index] (.opaque 2 [2,3] [(.bits 8)] (.opaque 3 [4,5] [(.bits 8)] (.ret [0]))))))
def constantAt (identity : Nat) : Option Int :=
  if identity = 0 then some 2 else
  if identity = 1 then some 3 else
  none
def model : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := indexIterations
  effect := fun identity arguments world =>
    if identity = 2 then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => (constantAt identity).getD (arguments.headD 0)),
      if (constantAt identity).isSome then world else world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_skips_body : evalProgram model source [3,2] 7 = .ok ([3],8) := rfl
theorem target_executes_body : evalProgram model target [3,2] 7 = .error (.external 17,8) := rfl
theorem observations_differ : evalProgram model source [3,2] 7 ≠ evalProgram model target [3,2] 7 := by
  rw [source_skips_body,target_executes_body]
  intro equal
  cases equal
end Merlin.GraphInlineEmptyCountermodel
