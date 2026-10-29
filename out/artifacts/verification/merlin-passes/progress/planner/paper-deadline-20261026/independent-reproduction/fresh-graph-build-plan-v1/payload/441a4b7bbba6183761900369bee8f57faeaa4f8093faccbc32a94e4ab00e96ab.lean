import Merlin.GraphStateIntegerOperations
import Merlin.GraphStateGateEntries
import Merlin.GraphStateScalar
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphGateChain.Boundary0.Scalar
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
namespace F1
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [5,6] (.pure (.add 8) [6,7] (.pure (.mul 8) [1,0] (.pure (.add 8) [4,0] (.opaque 3 [0] [(.bits 8)] (.ret [0])))))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
def after : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [5,6] (.pure (.mul 8) [0,0] (.pure (.add 8) [3,0] (.opaque 3 [0] [(.bits 8)] (.ret [0]))))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
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
end Merlin.GraphGateChain.Boundary0.Scalar
namespace Merlin.GraphGateChain.Boundary0.Entries
open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries
abbrev registry := Merlin.GraphGateChain.Boundary0.Scalar.registry
def beforeEntries : List Entry := [{inputs := Merlin.GraphGateChain.Boundary0.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Boundary0.Scalar.F1.outputs, body := Merlin.GraphGateChain.Boundary0.Scalar.F1.before},{inputs := Merlin.GraphGateChain.Boundary0.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Boundary0.Scalar.F2.outputs, body := Merlin.GraphGateChain.Boundary0.Scalar.F2.before}]
def afterEntries : List Entry := [{inputs := Merlin.GraphGateChain.Boundary0.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Boundary0.Scalar.F1.outputs, body := Merlin.GraphGateChain.Boundary0.Scalar.F1.after},{inputs := Merlin.GraphGateChain.Boundary0.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Boundary0.Scalar.F2.outputs, body := Merlin.GraphGateChain.Boundary0.Scalar.F2.after}]
theorem complete {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model beforeEntries afterEntries :=
  entries_cons model rfl rfl (Merlin.GraphGateChain.Boundary0.Scalar.F1.actual Value World Error model) (entries_cons model rfl rfl (Merlin.GraphGateChain.Boundary0.Scalar.F2.actual Value World Error model) (entries_nil model))
end Merlin.GraphGateChain.Boundary0.Entries
theorem merlin_chain_boundary0 {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphGateChain.Boundary0.Entries.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model
      Merlin.GraphGateChain.Boundary0.Entries.beforeEntries Merlin.GraphGateChain.Boundary0.Entries.afterEntries :=
  Merlin.GraphGateChain.Boundary0.Entries.complete model
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphGateChain.Boundary1.Scalar
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
namespace F1
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [5,6] (.pure (.mul 8) [0,0] (.pure (.add 8) [3,0] (.opaque 3 [0] [(.bits 8)] (.ret [0]))))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
def after : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.pure (.add 8) [3,4] (.pure (.mul 8) [0,0] (.loop 4 3 2 [5] [(.bits 8)] (.pure (.add 8) [1,2] (.opaque 3 [0] [(.bits 8)] (.ret [0]))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))))
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
end Merlin.GraphGateChain.Boundary1.Scalar
namespace Merlin.GraphGateChain.Boundary1.Entries
open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries
abbrev registry := Merlin.GraphGateChain.Boundary1.Scalar.registry
def beforeEntries : List Entry := [{inputs := Merlin.GraphGateChain.Boundary1.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Boundary1.Scalar.F1.outputs, body := Merlin.GraphGateChain.Boundary1.Scalar.F1.before},{inputs := Merlin.GraphGateChain.Boundary1.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Boundary1.Scalar.F2.outputs, body := Merlin.GraphGateChain.Boundary1.Scalar.F2.before}]
def afterEntries : List Entry := [{inputs := Merlin.GraphGateChain.Boundary1.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Boundary1.Scalar.F1.outputs, body := Merlin.GraphGateChain.Boundary1.Scalar.F1.after},{inputs := Merlin.GraphGateChain.Boundary1.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Boundary1.Scalar.F2.outputs, body := Merlin.GraphGateChain.Boundary1.Scalar.F2.after}]
theorem complete {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model beforeEntries afterEntries :=
  entries_cons model rfl rfl (Merlin.GraphGateChain.Boundary1.Scalar.F1.actual Value World Error model) (entries_cons model rfl rfl (Merlin.GraphGateChain.Boundary1.Scalar.F2.actual Value World Error model) (entries_nil model))
end Merlin.GraphGateChain.Boundary1.Entries
theorem merlin_chain_boundary1 {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphGateChain.Boundary1.Entries.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model
      Merlin.GraphGateChain.Boundary1.Entries.beforeEntries Merlin.GraphGateChain.Boundary1.Entries.afterEntries :=
  Merlin.GraphGateChain.Boundary1.Entries.complete model
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace Merlin.GraphGateChain.Endpoint.Scalar
open Merlin.GraphState.Scalar
def registry : List Signature := [{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [],outputs := [.index]},{inputs := [(.bits 8)],outputs := [(.bits 8)]}]
namespace F1
def inputs : List Ty := [(.bits 8),(.bits 8)]
def outputs : List Ty := [(.bits 8)]
def before : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.loop 2 1 0 [3] [(.bits 8)] (.pure (.add 8) [5,6] (.pure (.add 8) [6,7] (.pure (.mul 8) [1,0] (.pure (.add 8) [4,0] (.opaque 3 [0] [(.bits 8)] (.ret [0])))))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))
def after : Program := (.opaque 0 [] [.index] (.opaque 2 [] [.index] (.opaque 1 [] [.index] (.pure (.add 8) [3,4] (.pure (.mul 8) [0,0] (.loop 4 3 2 [5] [(.bits 8)] (.pure (.add 8) [1,2] (.opaque 3 [0] [(.bits 8)] (.ret [0]))) (.opaque 3 [0] [(.bits 8)] (.ret [0]))))))))
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
end Merlin.GraphGateChain.Endpoint.Scalar
namespace Merlin.GraphGateChain.Endpoint.Entries
open Merlin.GraphState.Scalar Merlin.GraphState.GateEntries
abbrev registry := Merlin.GraphGateChain.Endpoint.Scalar.registry
def beforeEntries : List Entry := [{inputs := Merlin.GraphGateChain.Endpoint.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Endpoint.Scalar.F1.outputs, body := Merlin.GraphGateChain.Endpoint.Scalar.F1.before},{inputs := Merlin.GraphGateChain.Endpoint.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Endpoint.Scalar.F2.outputs, body := Merlin.GraphGateChain.Endpoint.Scalar.F2.before}]
def afterEntries : List Entry := [{inputs := Merlin.GraphGateChain.Endpoint.Scalar.F1.inputs, outputs := Merlin.GraphGateChain.Endpoint.Scalar.F1.outputs, body := Merlin.GraphGateChain.Endpoint.Scalar.F1.after},{inputs := Merlin.GraphGateChain.Endpoint.Scalar.F2.inputs, outputs := Merlin.GraphGateChain.Endpoint.Scalar.F2.outputs, body := Merlin.GraphGateChain.Endpoint.Scalar.F2.after}]
theorem complete {Value World Error : Type} (model : Model Value World Error registry) :
    PreservesEntries model beforeEntries afterEntries :=
  entries_cons model rfl rfl (Merlin.GraphGateChain.Endpoint.Scalar.F1.actual Value World Error model) (entries_cons model rfl rfl (Merlin.GraphGateChain.Endpoint.Scalar.F2.actual Value World Error model) (entries_nil model))
end Merlin.GraphGateChain.Endpoint.Entries
theorem merlin_chain_endpoint {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphGateChain.Endpoint.Entries.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model
      Merlin.GraphGateChain.Endpoint.Entries.beforeEntries Merlin.GraphGateChain.Endpoint.Entries.afterEntries :=
  Merlin.GraphGateChain.Endpoint.Entries.complete model
theorem merlin_chain_registry_join : Merlin.GraphGateChain.Boundary0.Entries.registry = Merlin.GraphGateChain.Boundary1.Entries.registry := rfl
theorem merlin_chain_endpoint_registry : Merlin.GraphGateChain.Boundary0.Entries.registry = Merlin.GraphGateChain.Endpoint.Entries.registry := rfl
theorem merlin_chain_middle_join : Merlin.GraphGateChain.Boundary0.Entries.afterEntries = Merlin.GraphGateChain.Boundary1.Entries.beforeEntries := rfl
theorem merlin_chain_original_join : Merlin.GraphGateChain.Endpoint.Entries.beforeEntries = Merlin.GraphGateChain.Boundary0.Entries.beforeEntries := rfl
theorem merlin_chain_final_join : Merlin.GraphGateChain.Endpoint.Entries.afterEntries = Merlin.GraphGateChain.Boundary1.Entries.afterEntries := rfl
theorem merlin_chain_composed {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphGateChain.Endpoint.Entries.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model Merlin.GraphGateChain.Endpoint.Entries.beforeEntries Merlin.GraphGateChain.Endpoint.Entries.afterEntries := by
  intro identity arguments world
  exact (Merlin.GraphGateChain.Boundary0.Entries.complete model identity arguments world).trans
    (Merlin.GraphGateChain.Boundary1.Entries.complete model identity arguments world)
theorem merlin_chain_transform {Value World Error : Type}
    (model : Merlin.GraphState.Scalar.Model Value World Error Merlin.GraphGateChain.Endpoint.Entries.registry) :
    Merlin.GraphState.GateEntries.PreservesEntries model Merlin.GraphGateChain.Endpoint.Entries.beforeEntries Merlin.GraphGateChain.Endpoint.Entries.afterEntries :=
  Merlin.GraphGateChain.Endpoint.Entries.complete model
namespace Merlin.GraphGateChain.Witness
open Merlin.GraphState.Scalar
abbrev registry := Merlin.GraphGateChain.Endpoint.Entries.registry
def constants : List (Option Int) := [some (0),some (1),some (3),none]
def chosen (fail : Bool) : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun lower upper stride =>
    if 0 < stride then
      (List.range ((upper - lower + stride - 1) / stride).toNat).map
        (fun k => lower + (k : Int) * stride)
    else []
  effect := fun identity arguments world =>
    if (constants[identity]?.getD none).isSome then
      .ok ((signature registry identity).outputs.map
        (fun _ => (constants[identity]?.getD none).getD 0),world)
    else if fail then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success; simp only [List.length_map]
    · split at success
      · cases success
      · cases success; simp only [List.length_map]
theorem finite_control : (chosen false).iterations 0 3 1 = [0,1,2] := rfl
theorem source_success : evalProgram (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [3,5] 7 = .ok ([195],11) := rfl
theorem final_success : evalProgram (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [3,5] 7 = .ok ([195],11) := by
  rw [← Merlin.GraphGateChain.Endpoint.Scalar.F1.actual Int Nat Int (chosen false) [3,5] 7 rfl]
  exact source_success
theorem source_success_initialized : Initialized (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [3,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen false) [3,5] 7 rfl).1
theorem final_success_initialized : Initialized (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [3,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen false) [3,5] 7 rfl).2
theorem source_success_other : evalProgram (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [4,5] 7 = .ok ([247],11) := rfl
theorem final_success_other : evalProgram (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [4,5] 7 = .ok ([247],11) := by
  rw [← Merlin.GraphGateChain.Endpoint.Scalar.F1.actual Int Nat Int (chosen false) [4,5] 7 rfl]
  exact source_success_other
theorem source_success_other_initialized : Initialized (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [4,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen false) [4,5] 7 rfl).1
theorem final_success_other_initialized : Initialized (chosen false) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [4,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen false) [4,5] 7 rfl).2
theorem source_error : evalProgram (chosen true) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [3,5] 7 = .error (.external 17,8) := rfl
theorem final_error : evalProgram (chosen true) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [3,5] 7 = .error (.external 17,8) := by
  rw [← Merlin.GraphGateChain.Endpoint.Scalar.F1.actual Int Nat Int (chosen true) [3,5] 7 rfl]
  exact source_error
theorem source_error_initialized : Initialized (chosen true) Merlin.GraphGateChain.Endpoint.Scalar.F1.before [3,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen true) [3,5] 7 rfl).1
theorem final_error_initialized : Initialized (chosen true) Merlin.GraphGateChain.Endpoint.Scalar.F1.after [3,5] 7 :=
  (Merlin.GraphGateChain.Endpoint.Scalar.F1.operands_initialized (chosen true) [3,5] 7 rfl).2
end Merlin.GraphGateChain.Witness
