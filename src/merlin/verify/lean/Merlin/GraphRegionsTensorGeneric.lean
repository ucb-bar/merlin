/- Full tensor execution and poison-initializer independence, reusing the frozen
parallel full-overwrite theorem. Concrete scalar meanings come from Core/Sem. -/
import Merlin.GraphRegionsTensorBody

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Sem.Ops.Linalg

def InputsInitialized (types : ScalarTypes) (shape : List Nat) (inputs : List Tensor) : Prop :=
  List.Forall₂ (fun pair tensor => TensorInitialized shape pair.2 tensor) types inputs

def pointValues (inputs : List Tensor) (point : Idx) : List Nat :=
  inputs.map fun tensor => valueNat (tensor.get point)

def Body.point (body : Body) (inputs : List Tensor) (point : Idx) : Val :=
  .some (.bits body.width (BitVec.ofNat body.width (body.result (pointValues inputs point))))

def Body.tensor (body : Body) (shape : List Nat) (inputs : List Tensor) : Tensor :=
  Tensor.canon shape (.int body.width) (body.point inputs)

theorem inputs_point {types : ScalarTypes} {shape point : List Nat} {inputs : List Tensor}
    (initialized : InputsInitialized types shape inputs) (bounds : Idx.InBounds shape point) :
    inputValues types (pointValues inputs point) = inputs.map (fun tensor => .scalar (tensor.get point)) := by
  induction initialized with
  | nil => rfl
  | @cons pair tensor types inputs head tail ih =>
    simp only [pointValues, List.map_cons, inputValues]
    rw [extracted_scalar (head.2.2 point bounds)]
    exact congrArg _ ih

theorem read_inputs (inputs : List Tensor) (shape point : List Nat)
    (shapes : ∀ tensor ∈ inputs, tensor.shape = shape) (bounds : Idx.InBounds shape point) :
    ((inputs.map RVal.tensor).zip (List.replicate inputs.length (identityMap shape.length))).mapM
      (fun (value, map) => readAt value map point) = .ok (inputs.map (·.get point)) := by
  induction inputs with
  | nil => rfl
  | cons tensor inputs ih =>
    have shapeHead := shapes tensor (by simp)
    have read : readAt (.tensor tensor) (identityMap shape.length) point = .ok (tensor.get point) :=
      by simpa [shapeHead] using readAt_identity (t := tensor) (shapeHead ▸ bounds)
    simp only [List.map_cons, List.length_cons, List.replicate_succ, List.zip_cons_cons,
      List.mapM_cons, List.mapM_nil, Outcome.bindOp_eq, Outcome.pure_eq]
    rw [read]
    simp only [Outcome.bind_ok]
    rw [ih (fun tensor member => shapes tensor (by simp [member]))]
    rfl

theorem input_shapes {types : ScalarTypes} {shape : List Nat} {inputs : List Tensor}
    (initialized : InputsInitialized types shape inputs) :
    ∀ tensor ∈ inputs, tensor.shape = shape := by
  induction initialized with
  | nil => simp
  | @cons pair tensor types inputs head tail ih =>
    intro value member
    rcases List.mem_cons.mp member with rfl | member
    · exact head.1
    · exact ih value member

theorem generic_point {body : Body} (valid : body.Valid) {shape : List Nat} {inputs : List Tensor}
    (initialized : InputsInitialized body.inputs shape inputs) (init : Tensor) (initShape : init.shape = shape)
    (point : Idx) (bounds : Idx.InBounds shape point) (ctx : Ctx) (call : CallSem) (environment : Env)
    (captures : ScalarAgree (constantTypes body.captures) environment (constantState body.captures)) :
    bodyAt ctx (inputs.map RVal.tensor) (List.replicate inputs.length (identityMap shape.length))
      [identityMap shape.length] [init]
      (fun nested => evalRegion opSem nested call environment body.region) point =
      .ok [body.point inputs point] := by
  have arity : (pointValues inputs point).length = body.inputs.length :=
    by simpa [pointValues] using initialized.length_eq.symm
  have execution := body_execution valid (pointValues inputs point) arity (init.get point)
    {ctx with loopIndex := point} call environment captures
  rw [inputs_point initialized bounds] at execution
  have readOld : readAt (.tensor init) (identityMap shape.length) point = .ok (init.get point) :=
    by simpa [initShape] using readAt_identity (t := init) (initShape ▸ bounds)
  unfold bodyAt
  rw [read_inputs inputs shape point (input_shapes initialized) bounds]
  simp only [Outcome.bindOp_eq, Outcome.pure_eq, Outcome.bind_ok, List.zip_cons_cons,
    List.zip_nil_right, List.mapM_cons, List.mapM_nil, readOld]
  simp only [Outcome.bind_ok, List.map_append, List.map_cons, List.map_nil, List.map_map, Function.comp_def]
  rw [execution]
  rfl

/-- Complete overwrite succeeds independently of every old destination element,
including all poison from tensor.empty. Initialized output follows constructively. -/
theorem generic_execution {body : Body} (valid : body.Valid) {shape : List Nat} {inputs : List Tensor}
    (initialized : InputsInitialized body.inputs shape inputs) (init : Tensor) (initShape : init.shape = shape)
    (initElement : init.elt = .int body.width) (ctx : Ctx) (call : CallSem) (environment : Env)
    (captures : ScalarAgree (constantTypes body.captures) environment (constantState body.captures)) :
    genericSem ctx (List.replicate (inputs.length + 1) (identityMap shape.length))
      (List.replicate shape.length .parallel) (inputs.map RVal.tensor) [.tensor init]
      (fun nested => evalRegion opSem nested call environment body.region) =
      .ok [.tensor (body.tensor shape inputs)] := by
  unfold Body.tensor
  rw [← initElement]
  apply MerlinVerify.Rules.ParallelFusion.identity_generic
  · intro tensor member
    rcases List.mem_append.mp member with input | output
    · exact input_shapes initialized tensor input
    · have equal : tensor = init := List.mem_singleton.mp output
      simpa [equal] using initShape
  · intro point bounds
    exact generic_point valid initialized init initShape point bounds ctx call environment captures

theorem tensor_result_initialized (body : Body) (shape : List Nat) (inputs : List Tensor) :
    TensorInitialized shape body.width (body.tensor shape inputs) := by
  refine ⟨rfl, rfl, ?_⟩
  intro point bounds
  refine ⟨BitVec.ofNat body.width (body.result (pointValues inputs point)), ?_⟩
  simp [Body.tensor, Tensor.canon, Body.point, bounds]

end Merlin.GraphRegionsSource
