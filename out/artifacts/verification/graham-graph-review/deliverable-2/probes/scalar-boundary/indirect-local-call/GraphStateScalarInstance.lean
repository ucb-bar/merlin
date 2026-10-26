import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateScalarInstance
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [(.opaque 0),(.bits 8)],outputs := [(.bits 8)]},{inputs := [],outputs := [(.opaque 0)]}]
namespace F0
def inputs : List Ty := [(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.pure (.sub 8) [0,0] (.ret [0]))
def after : Program := (.pure (.sub 8) [0,0] (.ret [0]))
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
namespace F1
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 1 [] [(.opaque 0)] (.pure (.add 8) [1,2] (.pure (.add 8) [2,3] (.pure (.mul 8) [1,0] (.opaque 0 [3,0] [(.bits 8)] (.ret [0]))))))
def after : Program := (.opaque 1 [] [(.opaque 0)] (.pure (.add 8) [1,2] (.pure (.mul 8) [0,0] (.opaque 0 [2,0] [(.bits 8)] (.ret [0])))))
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
end F1
end Merlin.GraphStateScalarInstance
