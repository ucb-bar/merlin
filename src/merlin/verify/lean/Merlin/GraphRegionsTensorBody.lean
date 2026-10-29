/- Scalar block binding and pointwise tensor initialization, using the unchanged
MerlinVerify values/evaluator. An old output element may be arbitrary poison. -/
import Merlin.GraphRegionsSourceBody
import Merlin.GraphRegionsSourceConstants
import MerlinVerify.Rules.ParallelFusion

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Sem.Ops.Linalg

def inputValues : ScalarTypes → List Nat → List RVal
  | (_, width) :: types, value :: values => scalar width value :: inputValues types values
  | _, _ => []

def inputEnvironment : ScalarTypes → List Nat → Env → Env
  | (id, width) :: types, value :: values, environment =>
    (id, scalar width value) :: inputEnvironment types values environment
  | _, _, environment => environment

def seedWith : ScalarTypes → List Nat → ScalarState → ScalarState
  | (id, _) :: types, value :: values, base => put (seedWith types values base) id value
  | _, _, base => base

def seed (types : ScalarTypes) (values : List Nat) : ScalarState := seedWith types values (fun _ => 0)

theorem input_agrees (types : ScalarTypes) (values : List Nat) (arity : values.length = types.length)
    (environment : Env) : ScalarAgree types (inputEnvironment types values environment) (seed types values) := by
  induction types generalizing values with
  | nil => intro id width found; simp at found
  | cons pair types ih =>
    obtain ⟨id, width⟩ := pair
    cases values with
    | nil => simp at arity
    | cons value values =>
      have tail : values.length = types.length := by simpa using arity
      exact agree_put (ih values tail) id width value

theorem input_agrees_with (types : ScalarTypes) (values : List Nat) (arity : values.length = types.length)
    (environment : Env) (base : ScalarState) (capturedTypes : ScalarTypes)
    (captured : ScalarAgree capturedTypes environment base) :
    ScalarAgree (types ++ capturedTypes) (inputEnvironment types values environment) (seedWith types values base) := by
  induction types generalizing values with
  | nil => exact captured
  | cons pair types ih =>
    obtain ⟨id, width⟩ := pair
    cases values with
    | nil => simp at arity
    | cons value values => exact agree_put (ih values (by simpa using arity)) id width value

def inputArguments (types : ScalarTypes) : List Value :=
  types.map fun (id, width) => ⟨id, .int width⟩

theorem bind_inputs (types : ScalarTypes) (values : List Nat) (arity : values.length = types.length)
    (old : Value) (previous : Val) (environment : Env) :
    environment.bindAll (inputArguments types ++ [old]) (inputValues types values ++ [.scalar previous]) =
      .ok (inputEnvironment types values ((old.id, .scalar previous) :: environment)) := by
  have lengths : (inputValues types values).length = types.length := by
    induction types generalizing values with
    | nil => simp [inputValues]
    | cons pair types ih =>
      cases values with
      | nil => simp at arity
      | cons value values => simpa [inputValues] using ih values (by simpa using arity)
  simp only [Env.bindAll, List.length_append, inputArguments, List.length_map, lengths,
    List.length_singleton, ↓reduceIte]
  congr 1
  induction types generalizing values with
  | nil => simp [inputValues, inputEnvironment]
  | cons pair types ih =>
    cases values with
    | nil => simp at arity
    | cons value values =>
      simp only [List.map_cons, List.cons_append, inputValues, List.zip_cons_cons,
        inputEnvironment]
      congr 1
      apply ih values (by simpa using arity)
      simpa [inputValues] using lengths

structure Body where
  inputs : ScalarTypes
  old : Nat
  width : Nat
  instructions : List Instruction
  returned : Nat
  captures : List CapturedConstant := []
  deriving DecidableEq, Repr

def Body.region (body : Body) : Region :=
  .mk [.mk (inputArguments body.inputs ++ [⟨body.old, .int body.width⟩])
    (body.instructions.map Instruction.operation ++ [yieldOp body.returned])]

def Body.Valid (body : Body) : Prop :=
  (body.inputs.map Prod.fst).Nodup ∧ body.old ∉ body.inputs.map Prod.fst ∧
    ScalarValid (body.inputs ++ constantTypes body.captures) body.instructions ∧
    (finalTypes body.instructions (body.inputs ++ constantTypes body.captures)).lookup body.returned = some body.width ∧
    body.old ∉ (constantTypes body.captures).map Prod.fst ∧
    (body.captures.map CapturedConstant.id).Nodup ∧ ∀ constant ∈ body.captures, 0 < constant.width

instance (body : Body) : Decidable body.Valid := by
  unfold Body.Valid
  infer_instance

def Body.result (body : Body) (values : List Nat) : Nat :=
  runScalar body.instructions (seedWith body.inputs values (constantState body.captures)) body.returned

/-- Actual block binding and all scalar operations execute. The previous output
is unconstrained and cannot enter the checked scalar type environment. -/
theorem body_execution {body : Body} (valid : body.Valid) (values : List Nat)
    (arity : values.length = body.inputs.length) (previous : Val)
    (ctx : Ctx) (call : CallSem) (environment : Env)
    (captures : ScalarAgree (constantTypes body.captures) environment (constantState body.captures)) :
    evalRegion opSem ctx call environment body.region (inputValues body.inputs values ++ [.scalar previous]) =
      .ok [scalar body.width (body.result values)] := by
  simp only [Body.region, evalRegion, evalBlocks, evalBlock]
  rw [bind_inputs _ _ arity, Outcome.bind_ok]
  have closure := scalar_agree_fresh captures valid.2.2.2.2.1 (.scalar previous)
  exact scalar_body_execution valid.2.2.1 (input_agrees_with _ _ arity _ _ _ closure) valid.2.2.2.1 ctx call

def valueNat : Val → Nat
  | .some (.bits _ bits) => bits.toNat
  | _ => 0

/-- Initializedness includes the exact integer width, not merely nonpoison. -/
def Initialized (width : Nat) (value : Val) : Prop :=
  ∃ bits : BitVec width, value = .some (.bits width bits)

theorem extracted_scalar {width : Nat} {value : Val} (initialized : Initialized width value) :
    scalar width (valueNat value) = .scalar value := by
  obtain ⟨bits, rfl⟩ := initialized
  exact scalar_of_bits bits

def TensorInitialized (shape : List Nat) (width : Nat) (tensor : Tensor) : Prop :=
  tensor.shape = shape ∧ tensor.elt = .int width ∧
    ∀ point, Idx.InBounds shape point → Initialized width (tensor.get point)

theorem initialized_typed {shape : List Nat} {width : Nat} {tensor : Tensor}
    (initialized : TensorInitialized shape width tensor) (ib : Nat) :
    RVal.HasTy ib (.tensor tensor) (.tensor shape (.int width)) := by
  refine ⟨initialized.1, initialized.2.1, ?_⟩
  intro point bounds
  obtain ⟨bits, found⟩ := initialized.2.2 point bounds
  simp [found, Val.HasTy, Scalar.conforms]

theorem initialized_defined {shape : List Nat} {width : Nat} {tensor : Tensor}
    (initialized : TensorInitialized shape width tensor) : (RVal.tensor tensor).Defined := by
  intro point bounds
  obtain ⟨bits, found⟩ := initialized.2.2 point (initialized.1 ▸ bounds)
  simp [found]

end Merlin.GraphRegionsSource
