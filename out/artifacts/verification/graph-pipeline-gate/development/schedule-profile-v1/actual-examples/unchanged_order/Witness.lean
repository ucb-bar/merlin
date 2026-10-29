import Merlin.GraphStateIntegerOperations
import Merlin.GraphStateGateEntries
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphStateScalarInstance
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [(.bits 8)],outputs := [(.bits 8)]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
namespace F2
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.pure (.add 8) [0,1] (.pure (.mul 8) [1,2] (.opaque 0 [1] [(.bits 8)] (.pure (.sub 8) [3,4] (.opaque 1 [2] [(.bits 8)] (.pure (.add 8) [2,0] (.pure (.add 8) [0,2] (.ret [0]))))))))
def after : Program := (.pure (.add 8) [0,1] (.pure (.mul 8) [1,2] (.opaque 0 [1] [(.bits 8)] (.pure (.sub 8) [3,4] (.opaque 1 [2] [(.bits 8)] (.pure (.add 8) [2,0] (.pure (.add 8) [0,2] (.ret [0]))))))))
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
namespace F3
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
end F3
end Merlin.GraphStateScalarInstance
namespace Merlin.GraphPipelineGateInstance
open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries
abbrev registry := Merlin.GraphStateScalarInstance.registry
def beforeEntries : List Entry := [{inputs := Merlin.GraphStateScalarInstance.F2.inputs, outputs := Merlin.GraphStateScalarInstance.F2.outputs, body := Merlin.GraphStateScalarInstance.F2.before},{inputs := Merlin.GraphStateScalarInstance.F3.inputs, outputs := Merlin.GraphStateScalarInstance.F3.outputs, body := Merlin.GraphStateScalarInstance.F3.before}]
def afterEntries : List Entry := [{inputs := Merlin.GraphStateScalarInstance.F2.inputs, outputs := Merlin.GraphStateScalarInstance.F2.outputs, body := Merlin.GraphStateScalarInstance.F2.after},{inputs := Merlin.GraphStateScalarInstance.F3.inputs, outputs := Merlin.GraphStateScalarInstance.F3.outputs, body := Merlin.GraphStateScalarInstance.F3.after}]
theorem complete {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model beforeEntries afterEntries :=
  entries_cons model rfl rfl (Merlin.GraphStateScalarInstance.F2.actual Value World Error model) (entries_cons model rfl rfl (Merlin.GraphStateScalarInstance.F3.actual Value World Error model) (entries_nil model))
end Merlin.GraphPipelineGateInstance
theorem merlin_schedule_transform {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphPipelineGateInstance.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model
      Merlin.GraphPipelineGateInstance.beforeEntries Merlin.GraphPipelineGateInstance.afterEntries :=
  Merlin.GraphPipelineGateInstance.complete model
namespace Merlin.GraphScheduleWitness
open Merlin.GraphState.Scalar
abbrev registry := Merlin.GraphStateScalarInstance.registry
def chosen (failAt : Option Nat) : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity arguments world =>
    if failAt = some identity then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_success : evalProgram (chosen none) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 = .ok ([21],9) := rfl
theorem target_success : evalProgram (chosen none) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 = .ok ([21],9) := by
  rw [← Merlin.GraphStateScalarInstance.F2.actual Int Nat Int (chosen none) [3,5] 7 rfl]
  exact source_success
theorem source_success_initialized : Initialized (chosen none) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen none) [3,5] 7 rfl).1
theorem target_success_initialized : Initialized (chosen none) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen none) [3,5] 7 rfl).2
theorem source_success_other : evalProgram (chosen none) Merlin.GraphStateScalarInstance.F2.before [4,5] 7 = .ok ([28],9) := rfl
theorem target_success_other : evalProgram (chosen none) Merlin.GraphStateScalarInstance.F2.after [4,5] 7 = .ok ([28],9) := by
  rw [← Merlin.GraphStateScalarInstance.F2.actual Int Nat Int (chosen none) [4,5] 7 rfl]
  exact source_success_other
theorem source_success_other_initialized : Initialized (chosen none) Merlin.GraphStateScalarInstance.F2.before [4,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen none) [4,5] 7 rfl).1
theorem target_success_other_initialized : Initialized (chosen none) Merlin.GraphStateScalarInstance.F2.after [4,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen none) [4,5] 7 rfl).2
theorem source_first_failure : evalProgram (chosen (some 0)) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 = .error (.external 17,8) := rfl
theorem target_first_failure : evalProgram (chosen (some 0)) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 = .error (.external 17,8) := by
  rw [← Merlin.GraphStateScalarInstance.F2.actual Int Nat Int (chosen (some 0)) [3,5] 7 rfl]
  exact source_first_failure
theorem source_first_failure_initialized : Initialized (chosen (some 0)) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen (some 0)) [3,5] 7 rfl).1
theorem target_first_failure_initialized : Initialized (chosen (some 0)) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen (some 0)) [3,5] 7 rfl).2
theorem source_second_failure : evalProgram (chosen (some 1)) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 = .error (.external 17,9) := rfl
theorem target_second_failure : evalProgram (chosen (some 1)) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 = .error (.external 17,9) := by
  rw [← Merlin.GraphStateScalarInstance.F2.actual Int Nat Int (chosen (some 1)) [3,5] 7 rfl]
  exact source_second_failure
theorem source_second_failure_initialized : Initialized (chosen (some 1)) Merlin.GraphStateScalarInstance.F2.before [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen (some 1)) [3,5] 7 rfl).1
theorem target_second_failure_initialized : Initialized (chosen (some 1)) Merlin.GraphStateScalarInstance.F2.after [3,5] 7 :=
  (Merlin.GraphStateScalarInstance.F2.operands_initialized (chosen (some 1)) [3,5] 7 rfl).2
end Merlin.GraphScheduleWitness
