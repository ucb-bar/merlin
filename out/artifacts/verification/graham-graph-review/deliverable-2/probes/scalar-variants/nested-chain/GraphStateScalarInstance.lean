import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateScalarInstance
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [.index,.index,(.bits 8)],outputs := []},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]}]
namespace F0
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.loop 4 3 2 [1] [(.bits 8)] (.pure (.mul 8) [7,8] (.pure (.mul 8) [8,9] (.opaque 0 [4,2,1] [] (.pure (.add 8) [3,0] (.ret [0]))))) (.ret [0])) (.ret [0])))))
def after : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.pure (.mul 8) [3,4] (.loop 3 2 1 [4] [(.bits 8)] (.loop 5 4 3 [1] [(.bits 8)] (.opaque 0 [2,0,4] [] (.pure (.add 8) [1,4] (.ret [0]))) (.ret [0])) (.ret [0]))))))
theorem accepted : check registry inputs outputs before after = true := by decide
theorem actual : Preserves registry inputs.length before after := check_sound accepted
theorem typed : Typed registry inputs outputs before ∧ Typed registry inputs outputs after :=
  check_typed accepted
theorem initialized {Value World Error : Type} (model : Model Value World Error registry)
    (values : List Value) (world : World) (arity : values.length = inputs.length) :
    Shaped outputs.length (evalProgram model before values world) ∧
    Shaped outputs.length (evalProgram model after values world) :=
  check_initialized accepted model values world arity
theorem operands_initialized {Value World Error : Type} (model : Model Value World Error registry)
    (values : List Value) (world : World) (arity : values.length = inputs.length) :
    Initialized model before values world ∧ Initialized model after values world :=
  check_operands_initialized accepted model values world arity
end F0
namespace F0
def componentBefore : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.loop 4 3 2 [1] [(.bits 8)] (.pure (.mul 8) [7,8] (.pure (.mul 8) [8,9] (.opaque 0 [4,2,1] [] (.pure (.add 8) [3,0] (.ret [0]))))) (.ret [0])) (.ret [0])))))
def middleLeft : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.loop 4 3 2 [1] [(.bits 8)] (.pure (.mul 8) [7,8] (.opaque 0 [3,1,0] [] (.pure (.add 8) [2,0] (.ret [0])))) (.ret [0])) (.ret [0])))))
def middleRight : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.loop 4 3 2 [1] [(.bits 8)] (.pure (.mul 8) [7,8] (.opaque 0 [3,1,0] [] (.pure (.add 8) [2,0] (.ret [0])))) (.ret [0])) (.ret [0])))))
def componentAfter : Program := (.opaque 1 [] [.index] (.opaque 3 [] [.index] (.opaque 2 [] [.index] (.pure (.mul 8) [3,4] (.loop 3 2 1 [4] [(.bits 8)] (.loop 5 4 3 [1] [(.bits 8)] (.opaque 0 [2,0,4] [] (.pure (.add 8) [1,4] (.ret [0]))) (.ret [0])) (.ret [0]))))))
theorem source_join : componentBefore = before := by decide
theorem middle_join : middleLeft = middleRight := by decide
theorem final_join : componentAfter = after := by decide
theorem first_accepted : check registry inputs outputs componentBefore middleLeft = true := by decide
theorem second_accepted : check registry inputs outputs middleRight componentAfter = true := by decide
theorem first_actual : Preserves registry inputs.length componentBefore middleLeft :=
  check_sound first_accepted
theorem second_actual : Preserves registry inputs.length middleRight componentAfter :=
  check_sound second_accepted
theorem middle_typed : Typed registry inputs outputs middleLeft := (check_typed first_accepted).2
theorem independently_composed : Preserves registry inputs.length before after := by
  have linked : Preserves registry inputs.length middleLeft componentAfter := by
    rw [middle_join]; exact second_actual
  simpa only [source_join,final_join] using preserves_trans first_actual linked
end F0
end Merlin.GraphStateScalarInstance
