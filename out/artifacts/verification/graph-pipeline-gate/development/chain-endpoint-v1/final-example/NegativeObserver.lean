import Merlin.GraphStateGateEntries
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateScalarInstance
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
namespace F1
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [5,6] (.pure (.add 8) [6,7] (.pure (.mul 8) [1,0] (.pure (.add 8) [4,0] (.opaque 3 [0] [(.bits 8)] (.ret [0])))))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
def after : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.pure (.add 8) [3,4] (.pure (.mul 8) [0,0] (.loop 4 3 2 [5] [(.bits 8)] (.pure (.add 8) [1,2] (.opaque 3 [0] [(.bits 8)] (.ret [0]))) (.opaque 3 [6] [(.bits 8)] (.ret [0]))))))))
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
namespace F2
def inputs : List Ty := [(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.ret [0])
def after : Program := (.ret [0])
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
end F2
end Merlin.GraphStateScalarInstance
namespace Merlin.GraphPipelineGateInstance
open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries
abbrev registry := Merlin.GraphStateScalarInstance.registry
def beforeEntries : List Entry := [{inputs := Merlin.GraphStateScalarInstance.F1.inputs, outputs := Merlin.GraphStateScalarInstance.F1.outputs, body := Merlin.GraphStateScalarInstance.F1.before},{inputs := Merlin.GraphStateScalarInstance.F2.inputs, outputs := Merlin.GraphStateScalarInstance.F2.outputs, body := Merlin.GraphStateScalarInstance.F2.before}]
def afterEntries : List Entry := [{inputs := Merlin.GraphStateScalarInstance.F1.inputs, outputs := Merlin.GraphStateScalarInstance.F1.outputs, body := Merlin.GraphStateScalarInstance.F1.after},{inputs := Merlin.GraphStateScalarInstance.F2.inputs, outputs := Merlin.GraphStateScalarInstance.F2.outputs, body := Merlin.GraphStateScalarInstance.F2.after}]
theorem complete {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model beforeEntries afterEntries :=
  entries_cons model rfl rfl (Merlin.GraphStateScalarInstance.F1.actual Value World Error model) (entries_cons model rfl rfl (Merlin.GraphStateScalarInstance.F2.actual Value World Error model) (entries_nil model))
end Merlin.GraphPipelineGateInstance
theorem merlin_transform {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphPipelineGateInstance.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model
      Merlin.GraphPipelineGateInstance.beforeEntries Merlin.GraphPipelineGateInstance.afterEntries :=
  Merlin.GraphPipelineGateInstance.complete model
